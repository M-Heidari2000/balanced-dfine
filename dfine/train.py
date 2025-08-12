import os
import json
import torch
import torch.nn as nn
import einops
import numpy as np
from pathlib import Path
from datetime import datetime
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard.writer import SummaryWriter
from .control_utils import compute_gramians
from .memory import ReplayBuffer
from .configs import TrainConfig
from .models import (
    Encoder,
    Decoder,
    Posterior,
    CostModel,
)

def train_backbone(
    config: TrainConfig,
    train_replay_buffer: ReplayBuffer,
    test_replay_buffer: ReplayBuffer,
):

    # prepare logging
    log_dir = Path(config.log_dir) / datetime.now().strftime("%Y%m%d_%H%M")
    os.makedirs(log_dir, exist_ok=True)
    with open(log_dir / "args.json", "w") as f:
        json.dump(config.dict(), f)
    
    writer = SummaryWriter(log_dir=log_dir)

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    # define models and optimizer
    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = Encoder(
        y_dim=train_replay_buffer.y_dim,
        a_dim=config.a_dim,
        hidden_dim=config.hidden_dim,
        dropout_p=config.dropout_p,
    ).to(device)

    decoder = Decoder(
        y_dim=train_replay_buffer.y_dim,
        a_dim=config.a_dim,
        hidden_dim=config.hidden_dim,
        dropout_p=config.dropout_p,
    ).to(device)

    posterior = Posterior(
        x_dim=config.x_dim,
        u_dim=train_replay_buffer.u_dim,
        a_dim=config.a_dim,
        device=device,
    ).to(device)

    all_params = (
        list(encoder.parameters()) +
        list(decoder.parameters()) + 
        list(posterior.parameters())
    )

    optimizer = torch.optim.Adam(all_params, lr=config.lr, eps=config.eps)

    # train and test loop
    for update in range(config.num_updates):

        # train
        encoder.train()
        decoder.train()
        posterior.train()

        y, u, c, _ = train_replay_buffer.sample(
            batch_size=config.batch_size,
            chunk_length=config.chunk_length,
        )

        # convert to tensor, transform to device, reshape to time-first
        y = torch.as_tensor(y, device=device)
        y = einops.rearrange(y, "b l y -> l b y")
        a = encoder(einops.rearrange(y, "l b y -> (l b) y"))
        a = einops.rearrange(a, "(l b) a -> l b a", b=config.batch_size)
        u = torch.as_tensor(u, device=device)
        u = einops.rearrange(u, "b l u -> l b u")

        # initial belief over x0: N(0, I)
        mean = torch.zeros((config.batch_size, config.x_dim), device=device)
        cov = torch.eye(config.x_dim, device=device).repeat([config.batch_size, 1, 1])

        y_pred_loss = 0

        for t in range(config.chunk_length - config.prediction_k - 1):
            mean, cov = posterior.dynamics_update(
                mean=mean,
                cov=cov,
                u=u[t],
            )
            mean, cov = posterior.measurement_update(
                mean=mean,
                cov=cov,
                a=a[t+1],
            )

            # tensors to hold predictions of future ys
            pred_y = torch.zeros((config.prediction_k, config.batch_size, train_replay_buffer.y_dim), device=device)
            pred_mean = mean
            pred_cov = cov

            for k in range(config.prediction_k):
                pred_mean, pred_cov = posterior.dynamics_update(
                    mean=pred_mean,
                    cov=pred_cov,
                    u=u[t+k+1]
                )
                pred_y[k] = decoder(pred_mean @ posterior.C.T)

            true_y = y[t+2:t+2+config.prediction_k]
            true_y_flatten = einops.rearrange(true_y, "k b y -> (k b) y")
            pred_y_flatten = einops.rearrange(pred_y, "k b y -> (k b) y")
            y_pred_loss += nn.MSELoss()(pred_y_flatten, true_y_flatten)

        y_pred_loss /= (config.chunk_length - config.prediction_k - 1)
        
        # y reconstruction loss
        y_flatten = einops.rearrange(y, "l b y -> (l b) y")
        a_flatten = einops.rearrange(a, "l b a -> (l b) a")
        y_recon = decoder(a_flatten)
        y_recon_loss = nn.MSELoss()(y_recon, y_flatten)

        # balancing loss
        Wc, Wo = compute_gramians(
            A=posterior.A,
            B=posterior.B,
            C=posterior.C
        )

        balancing_loss = 1 / torch.trace(Wc @ Wo)

        total_loss = (
            y_pred_loss +
            config.reconstruction_weight * y_recon_loss +
            config.balancing_weight * balancing_loss
        )

        optimizer.zero_grad()
        total_loss.backward()

        for name, param in posterior.named_parameters():
            print(f"{name}: {param.grad}")
        print("="*100)

        clip_grad_norm_(all_params, config.clip_grad_norm)
        optimizer.step()

        writer.add_scalar("train/ y prediction loss", y_pred_loss.item(), update)
        writer.add_scalar("train/ y reconstruction loss", y_recon_loss.item(), update)
        writer.add_scalar("train/ balancing loss", balancing_loss.item(), update)
        print(f"update step: {update+1}, train_loss: {total_loss.item()}")

        # test
        if update % config.test_interval == 0:
            # test
            encoder.eval()
            decoder.eval()
            posterior.eval()

            with torch.no_grad():

                y, u, c, _ = test_replay_buffer.sample(
                    batch_size=config.batch_size,
                    chunk_length=config.chunk_length,
                )

                # convert to tensor, transform to device, reshape to time-first
                y = torch.as_tensor(y, device=device)
                y = einops.rearrange(y, "b l y -> l b y")
                a = encoder(einops.rearrange(y, "l b y -> (l b) y"))
                a = einops.rearrange(a, "(l b) a -> l b a", b=config.batch_size)
                u = torch.as_tensor(u, device=device)
                u = einops.rearrange(u, "b l u -> l b u")

                # initial belief over x0: N(0, I)
                mean = torch.zeros((config.batch_size, config.x_dim), device=device)
                cov = torch.eye(config.x_dim, device=device).repeat([config.batch_size, 1, 1])

                y_pred_loss = 0

                for t in range(config.chunk_length - config.prediction_k - 1):
                    mean, cov = posterior.dynamics_update(
                        mean=mean,
                        cov=cov,
                        u=u[t],
                    )
                    mean, cov = posterior.measurement_update(
                        mean=mean,
                        cov=cov,
                        a=a[t+1],
                    )

                    # tensors to hold predictions of future ys
                    pred_y = torch.zeros((config.prediction_k, config.batch_size, train_replay_buffer.y_dim), device=device)

                    pred_mean = mean
                    pred_cov = cov

                    for k in range(config.prediction_k):
                        pred_mean, pred_cov = posterior.dynamics_update(
                            mean=pred_mean,
                            cov=pred_cov,
                            u=u[t+k+1]
                        )
                        pred_y[k] = decoder(pred_mean @ posterior.C.T)

                    true_y = y[t+2: t+2+config.prediction_k]
                    true_y_flatten = einops.rearrange(true_y, "k b y -> (k b) y")
                    pred_y_flatten = einops.rearrange(pred_y, "k b y -> (k b) y")
                    y_pred_loss += nn.MSELoss()(pred_y_flatten, true_y_flatten)

                y_pred_loss /= (config.chunk_length - config.prediction_k - 1)

                # y reconstruction loss
                y_flatten = einops.rearrange(y, "l b y -> (l b) y")
                a_flatten = einops.rearrange(a, "l b a -> (l b) a")
                y_recon = decoder(a_flatten)
                y_recon_loss = nn.MSELoss()(y_recon, y_flatten)
                
                # balancing loss
                Wc, Wo = compute_gramians(
                    A=posterior.A,
                    B=posterior.B,
                    C=posterior.C
                )

                balancing_loss = 1 / torch.trace(Wc @ Wo)

                total_loss = (
                    y_pred_loss +
                    config.reconstruction_weight * y_recon_loss +
                    config.balancing_weight * balancing_loss
                )

                writer.add_scalar("test/ y prediction loss", y_pred_loss.item(), update)
                writer.add_scalar("test/ y reconstruction loss", y_recon_loss.item(), update)
                writer.add_scalar("test/ balancing loss", balancing_loss.item(), update)
                print(f"test step: {update+1}, test_loss: {total_loss.item()}")

    torch.save(encoder.state_dict(), log_dir / "encoder.pth")
    torch.save(decoder.state_dict(), log_dir / "decoder.pth")
    torch.save(posterior.state_dict(), log_dir / "posterior.pth")

    return {"model_dir": log_dir}


def train_cost(
    backbone_dir: Path,
    train_replay_buffer: ReplayBuffer,
    test_replay_buffer: ReplayBuffer,
):

    with open(backbone_dir / "args.json", "r") as f:
        config = TrainConfig(**json.load(f))

    # prepare logging
    log_dir = backbone_dir
    writer = SummaryWriter(log_dir=log_dir)

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    # define models and optimizer
    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = Encoder(
        y_dim=train_replay_buffer.y_dim,
        a_dim=config.a_dim,
        hidden_dim=config.hidden_dim,
        dropout_p=config.dropout_p,
    ).to(device)

    decoder = Decoder(
        y_dim=train_replay_buffer.y_dim,
        a_dim=config.a_dim,
        hidden_dim=config.hidden_dim,
        dropout_p=config.dropout_p,
    ).to(device)

    posterior = Posterior(
        x_dim=config.x_dim,
        u_dim=train_replay_buffer.u_dim,
        a_dim=config.a_dim,
        device=device,
    ).to(device)

    cost_model = CostModel(
        x_dim=config.x_dim,
        u_dim=train_replay_buffer.u_dim,
        device=device
    )

    # load the backbone
    encoder.load_state_dict(torch.load(backbone_dir / "encoder.pth", weights_only=True))
    posterior.load_state_dict(torch.load(backbone_dir / "posterior.pth", weights_only=True))
    decoder.load_state_dict(torch.load(backbone_dir / "decoder.pth", weights_only=True))

    # freeze backbone models
    for p in encoder.parameters():
        p.requires_grad = False

    for p in decoder.parameters():
        p.requires_grad = False

    for p in posterior.parameters():
        p.requires_grad = False

    encoder.eval()
    decoder.eval()
    posterior.eval()

    all_params = (
        list(cost_model.parameters())
    )

    optimizer = torch.optim.Adam(all_params, lr=config.lr, eps=config.eps)

    # train and test loop
    for update in range(config.num_cost_updates):

        # train
        cost_model.train()

        y, u, c, _ = train_replay_buffer.sample(
            batch_size=config.batch_size,
            chunk_length=config.chunk_length,
        )

        # convert to tensor, transform to device, reshape to time-first
        y = torch.as_tensor(y, device=device)
        y = einops.rearrange(y, "b l y -> l b y")
        a = encoder(einops.rearrange(y, "l b y -> (l b) y"))
        a = einops.rearrange(a, "(l b) a -> l b a", b=config.batch_size)
        u = torch.as_tensor(u, device=device)
        u = einops.rearrange(u, "b l u -> l b u")
        c = torch.as_tensor(c, device=device)
        c = einops.rearrange(c, "b l 1 -> l b 1")

        # initial belief over x0: N(0, I)
        mean = torch.zeros((config.batch_size, config.x_dim), device=device)
        cov = torch.eye(config.x_dim, device=device).repeat([config.batch_size, 1, 1])

        cost_loss = 0.0

        for t in range(0, config.chunk_length-1):
            mean, cov = posterior.dynamics_update(
                mean=mean,
                cov=cov,
                u=u[t],
            )
            mean, cov = posterior.measurement_update(
                mean=mean,
                cov=cov,
                a=a[t+1],
            )

            cost_loss += nn.MSELoss()(cost_model(x=mean, u=u[t+1]), c[t])

        cost_loss /= (config.chunk_length - 1)
        total_loss = cost_loss

        optimizer.zero_grad()
        total_loss.backward()
        
        print(f"A: {cost_model.A.grad}")
        print(f"B: {cost_model.B.grad}")
        print(f"q: {cost_model.q.grad}")
        print("="*100)
        
        clip_grad_norm_(all_params, config.clip_grad_norm)
        optimizer.step()

        writer.add_scalar("train/ cost loss", cost_loss.item(), update)
        print(f"update step: {update+1}, train_loss: {total_loss.item()}")

        # test
        if update % config.test_interval == 0:
            # test
            cost_model.eval()

            y, u, c, _ = test_replay_buffer.sample(
                batch_size=config.batch_size,
                chunk_length=config.chunk_length,
            )

            # convert to tensor, transform to device, reshape to time-first
            y = torch.as_tensor(y, device=device)
            y = einops.rearrange(y, "b l y -> l b y")
            a = encoder(einops.rearrange(y, "l b y -> (l b) y"))
            a = einops.rearrange(a, "(l b) a -> l b a", b=config.batch_size)
            u = torch.as_tensor(u, device=device)
            u = einops.rearrange(u, "b l u -> l b u")
            c = torch.as_tensor(c, device=device)
            c = einops.rearrange(c, "b l 1 -> l b 1")

            # initial belief over x0: N(0, I)
            mean = torch.zeros((config.batch_size, config.x_dim), device=device)
            cov = torch.eye(config.x_dim, device=device).repeat([config.batch_size, 1, 1])

            cost_loss = 0.0

            for t in range(0, config.chunk_length-1):
                mean, cov = posterior.dynamics_update(
                    mean=mean,
                    cov=cov,
                    u=u[t],
                )
                mean, cov = posterior.measurement_update(
                    mean=mean,
                    cov=cov,
                    a=a[t+1],
                )

                cost_loss += nn.MSELoss()(cost_model(x=mean, u=u[t+1]), c[t])

            cost_loss /= (config.chunk_length - 1)
            total_loss = cost_loss

            writer.add_scalar("test/ cost loss", cost_loss.item(), update)
            print(f"test step: {update+1}, test_loss: {total_loss.item()}")

    torch.save(cost_model.state_dict(), log_dir / "cost_model.pth")

    return {"model_dir": log_dir}