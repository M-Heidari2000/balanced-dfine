import os
import gymnasium
import json
import torch
import torch.nn as nn
import einops
import numpy as np
from pathlib import Path
from datetime import datetime
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard.writer import SummaryWriter
from .memory import ReplayBuffer
from .configs import TrainConfig
from .agents import MPCAgent
from .models import (
    Encoder,
    Decoder,
    Posterior,
    CostModel,
)


def train(
    config: TrainConfig,
    env: gymnasium.Env,
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

    cost_model = CostModel(
        x_dim=config.x_dim,
        u_dim=env.action_space.shape[0],
        device=device,
        hidden_dim=config.hidden_dim,
    )

    all_params = (
        list(encoder.parameters()) +
        list(decoder.parameters()) + 
        list(posterior.parameters()) +
        list(cost_model.parameters())
    )

    optimizer = torch.optim.Adam(all_params, lr=config.lr, eps=config.eps)

    # agent
    agent = MPCAgent(
        encoder=encoder,
        posterior=posterior,
        cost_model=cost_model,
        planning_horizon=config.planning_horizon
    )

    # replay buffer
    buffer = ReplayBuffer(
        capacity=100000,
        y_dim=env.observation_space.shape[0],
        u_dim=env.action_space.shape[0],
    )

    # collect seed episodes
    for s in range(1, config.seed_episodes+1):
        obs, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action=action)
            done = terminated or truncated
            buffer.push(
                y=obs,
                u=action,
                c=-reward,
                done=done
            )
            obs = next_obs

    # train and test loop
    for episode in range(config.all_episodes):
        
        # model fit
        for s in range(config.collect_interval):
            encoder.train()
            decoder.train()
            posterior.train()
            cost_model.train()

            y, u, c, _ = buffer.sample(
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

            y_pred_loss = 0
            cost_loss = 0

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
                cost_loss += nn.MSELoss()(cost_model(x=mean, u=u[t+1]), c[t])

                # tensors to hold predictions of future ys
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
                y_pred_loss += nn.MSELoss()(pred_y_flatten, true_y_flatten)

            y_pred_loss /= (config.chunk_length - config.prediction_k - 1)

            # y reconstruction loss
            y_flatten = einops.rearrange(y, "l b y -> (l b) y")
            a_flatten = einops.rearrange(a, "l b a -> (l b) a")
            y_recon = decoder(a_flatten)
            y_recon_loss = nn.MSELoss()(y_recon, y_flatten)

            # cost loss
            cost_loss /= (config.chunk_length - 1)
            total_loss = cost_loss
        
            total_loss = (
                y_pred_loss +
                cost_loss +
                config.reconstruction_weight * y_recon_loss
            )

            optimizer.zero_grad()
            total_loss.backward()

            clip_grad_norm_(all_params, config.clip_grad_norm)
            optimizer.step()

            global_step = episode * config.collect_interval + s
            writer.add_scalar("y prediction loss", y_pred_loss.item(), global_step)
            writer.add_scalar("y reconstruction loss", y_recon_loss.item(), global_step)
            writer.add_scalar("cost loss", cost_loss.item(), global_step)

        # data collection
        with torch.no_grad():
            obs, info = env.reset()
            agent.reset()
            action = env.action_space.sample()
            done = False
            while not done:
                planned_actions = agent(y=obs, u=action, explore=True)
                action = planned_actions[0]
                next_obs, reward, terminated, truncated, _ = env.step(action=action)
                done = terminated or truncated
                buffer.push(
                    y=obs,
                    u=action,
                    c=-reward,
                    done=done
                )
                obs = next_obs

        # test
        if episode % config.test_interval == 0:
            encoder.eval()
            decoder.eval()
            posterior.eval()
            cost_model.eval()
            with torch.no_grad():
                obs, info = env.reset()
                agent.reset()
                action = env.action_space.sample()
                done = False
                total_reward = 0.0
                while not done:
                    planned_actions = agent(y=obs, u=action, explore=False)
                    action = planned_actions[0]
                    next_obs, reward, terminated, truncated, _ = env.step(action=action)
                    done = terminated or truncated
                    obs = next_obs
                    total_reward += reward
            
            print(f"episode: {episode}, total reward:{total_reward}")
            writer.add_scalar("total reward", total_reward, global_step=episode)


    torch.save(encoder.state_dict(), log_dir / "encoder.pth")
    torch.save(decoder.state_dict(), log_dir / "decoder.pth")
    torch.save(posterior.state_dict(), log_dir / "posterior.pth")
    torch.save(cost_model.state_dict(), log_dir / "cost_model.pth")

    return {"model_dir": log_dir}