from dataclasses import dataclass, asdict


@dataclass
class TrainConfig:
    seed: int = 0
    log_dir: str = "log"
    x_dim: int = 30
    a_dim: int = 100
    hidden_dim: int = 32
    min_var: float = 1e-2
    dropout_p: float=0.4
    seed_episodes: int = 5
    all_episodes: int = 100
    collect_interval: int = 10
    test_interval: int = 10
    chunk_length: int = 50
    prediction_k: int = 10
    batch_size: int = 64
    planning_horizon: int = 12
    action_noise: float = 0.3
    lr: float = 1e-3
    eps: float = 1e-8
    clip_grad_norm: int = 1000
    reconstruction_weight: float = 1.0
    balancing_weight: float = 1.0
    
    dict = asdict