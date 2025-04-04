import gymnasium as gym
from .memory import ReplayBuffer


def collect_data(
    env: gym.Env,
    num_episodes: int
):
    
    replay_buffer = ReplayBuffer(
        capacity=num_episodes * env.horizon,
        y_dim=env.observation_space.shape[0],
        u_dim=env.action_space.shape[0],
    )

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.push(
                y=obs,
                u=action,
                done=done,
            )
            obs = next_obs
    
    return replay_buffer