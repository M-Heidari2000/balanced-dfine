import os
import json
import torch
import einops
import numpy as np
import gymnasium as gym
from pathlib import Path
from datetime import datetime
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard.writer import SummaryWriter
from .memory import ReplayBuffer
from .configs import TrainConfig
from .models import (
    Encoder,
    Decoder,
    Posterior,
)
from .control_utils import (
    solve_discrete_lyapunov,
    compute_gramians,
)


def train(env: gym.Env, config: TrainConfig):

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

    # replay buffers
    train_replay_buffer = ReplayBuffer(
        capacity=config.buffer_capacity,
        y_dim=env.observation_space.shape[0],
        u_dim=env.action_space.shape[0],
    )

    test_replay_buffer = ReplayBuffer(
        capacity=config.buffer_capacity,
        y_dim=env.observation_space.shape[0],
        u_dim=env.action_space.shape[0],
    )

    # define models and optimizer
    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = Encoder(
        y_dim=env.observation_space.shape[0],
        a_dim=config.a_dim,
        hidden_dim=config.hidden_dim,
        dropout_p=config.dropout_p,
    ).to(device)

    decoder = Decoder(
        y_dim=env.observation_space.shape[0],
        a_dim=config.a_dim,
        hidden_dim=config.hidden_dim,
        dropout_p=config.dropout_p,
    ).to(device)

    posterior = Posterior(
        x_dim=config.x_dim,
        u_dim=env.action_space.shape[0],
        a_dim=config.a_dim,
        device=device,
    ).to(device)

    all_params = (
        list(encoder.parameters()) +
        list(decoder.parameters()) + 
        list(posterior.parameters())
    )

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(all_params, lr=config.lr, eps=config.eps)

    # collect training data
    for _ in range(config.num_train_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            train_replay_buffer.push(
                y=obs,
                u=action,
                done=done,
            )
            obs = next_obs
    # collect test data
    for _ in range(config.num_test_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            test_replay_buffer.push(
                y=obs,
                u=action,
                done=done,
            )
            obs = next_obs

    # train and test loop
    for update in range(config.num_updates):

        # train
        encoder.train()
        decoder.train()
        posterior.train()

        y, u, _ = train_replay_buffer.sample(
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

            # a tensor to hold predictions of future ys
            pred_y = torch.zeros((config.prediction_k, config.batch_size, env.observation_space.shape[0]), device=device)

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

            y_pred_loss += criterion(pred_y_flatten, true_y_flatten)

        y_pred_loss /= config.chunk_length - config.prediction_k - 1

        # balancing loss
        Wc, Wo = compute_gramians(
            A=posterior.A,
            B=posterior.B,
            C=posterior.C
        )

        balancing_loss = 1 / torch.trace(Wc @ Wo)
        total_loss = y_pred_loss + config.balancing_weight * balancing_loss

        optimizer.zero_grad()
        total_loss.backward()
        clip_grad_norm_(all_params, config.clip_grad_norm)
        optimizer.step()

        writer.add_scalar("y prediction loss train", y_pred_loss.item(), update)
        writer.add_scalar("balancing loss train", balancing_loss.item(), update)
        print(f"update step: {update+1}, train_loss: {total_loss.item()}")

        # test
        if update % config.test_interval == 0:
            # test
            encoder.eval()
            decoder.eval()
            posterior.eval()

            with torch.no_grad():

                y, u, _ = test_replay_buffer.sample(
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

                    # a tensor to hold predictions of future ys
                    pred_y = torch.zeros((config.prediction_k, config.batch_size, env.observation_space.shape[0]), device=device)

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

                    y_pred_loss += criterion(pred_y_flatten, true_y_flatten)

                y_pred_loss /= config.chunk_length - config.prediction_k - 1

                # balancing loss
                Wc, Wo = compute_gramians(
                    A=posterior.A,
                    B=posterior.B,
                    C=posterior.C
                )

                balancing_loss = 1 / torch.trace(Wc @ Wo)
                total_loss = y_pred_loss + config.balancing_weight * balancing_loss

                writer.add_scalar("y prediction loss test", y_pred_loss.item(), update)
                writer.add_scalar("balancing loss test", balancing_loss.item(), update)
                print(f"update step: {update+1}, test_loss: {total_loss.item()}")

    torch.save(encoder.state_dict(), log_dir / "encoder.pth")
    torch.save(decoder.state_dict(), log_dir / "decoder.pth")
    torch.save(posterior.state_dict(), log_dir / "posterior.pth")

    return {"model_dir": log_dir}