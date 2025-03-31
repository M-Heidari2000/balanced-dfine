import torch
import torch.nn as nn
from typing import Optional


class Encoder(nn.Module):
    """
        y_t -> a_t
    """

    def __init__(
        self,
        a_dim: int,
        y_dim: int,
        hidden_dim: Optional[int]=None,
        dropout_p: float=0.4,
    ):
        super().__init__()

        hidden_dim = hidden_dim if hidden_dim is not None else 2*y_dim

        self.mlp_layers = nn.Sequential(
            nn.Linear(y_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, a_dim),
        )

    def forward(self, y):
        return self.mlp_layers(y)
    

class Decoder(nn.Module):
    """
        a_t -> y_t
    """

    def __init__(
        self,
        a_dim: int,
        y_dim: int,
        hidden_dim: Optional[int]=None,
        dropout_p: float=0.4,
    ):
        super().__init__()

        hidden_dim = hidden_dim if hidden_dim is not None else 2*a_dim

        self.mlp_layers = nn.Sequential(
            nn.Linear(a_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, y_dim),
        )

    def forward(self, a):
        return self.mlp_layers(a)


class Posterior(nn.Module):
    
    """
        KF that obtains belief over x_{t+1} using belief of x_t, u_t, and y_{t+1}
    """

    def __init__(
        self,
        x_dim: int,
        u_dim: int,
        a_dim: int,
        device: str,
        min_var: float=1e-4,
    ):
        super().__init__()

        self.x_dim = x_dim
        self.u_dim = u_dim
        self.a_dim = a_dim
        self.device = device
        self._min_var = min_var

        # Dynamics matrices
        self.M = nn.Parameter(
            torch.eye(self.x_dim, device=self.device)
        )
        self.d = nn.Parameter(
            torch.randn(self.x_dim, device=self.device)
        )
        self.B = nn.Parameter(
            torch.randn(self.x_dim, self.u_dim, device=self.device),
        )
        self.C = nn.Parameter(
            torch.randn(self.a_dim, self.x_dim, device=self.device)
        )

        # Transition noise covariance (diagonal)
        self.nx = nn.Parameter(
            torch.randn(self.x_dim, device=self.device)
        )
        # Observation noise covariance (diagonal)
        self.na = nn.Parameter(
            torch.randn(self.a_dim, device=device)
        )

    @property
    def A(self):
        # constructing a stable A matrix
        # softplus ensures positive entries
        d = nn.functional.softplus(self.d)
        # QR decomposition to obtain a unitary matrix
        # why sign correction of the columns?
        Q, R = torch.linalg.qr(self.M, mode='reduced')
        Q = Q @ R.diagonal().sign().diag()        
        return d.sqrt().diag() @ Q @ (1 / (1+d).sqrt()).diag()

    def dynamics_update(
        self,
        mean,
        cov,
        u,
    ):
        """
            Single step dynamics update

            mean: b x
            cov: b x x
            u: b u
        """

        Nx = torch.diag(nn.functional.softplus(self.nx) + self._min_var)    # shape: x x
        next_mean = mean @ self.A.T + u @ self.B.T
        next_cov = self.A @ cov @ self.A.T + Nx

        return next_mean, next_cov
    
    def measurement_update(
        self,
        mean,
        cov,
        a,
    ):
        """
            Single step measurement update
        
            mean: b x
            cov: b x x
            a: b a
        """

        Na = torch.diag(nn.functional.softplus(self.na) + self._min_var)    # shape: a a

        K = cov @ self.C.T @ torch.linalg.pinv(self.C @ cov @ self.C.T + Na)
        next_mean = mean + ((a - mean @ self.C.T).unsqueeze(1) @ K.transpose(1, 2)).squeeze(1)
        next_cov = (torch.eye(self.x_dim, device=self.device) - K @ self.C) @ cov

        return next_mean, next_cov