import torch
import numpy as np
from mpc import mpc
from mpc.mpc import QuadCost, LinDx


class LQRAgent:
    """
        action planning by the LQR method
    """
    def __init__(
        self,
        encoder,
        posterior,
        cost_model,
        planning_horizon: int,
    ):
        self.encoder = encoder
        self.posterior = posterior
        self.cost_model = cost_model
        self.planning_horizon = planning_horizon

        self.device = next(encoder.parameters()).device
        self.Ks, self.ks = self._compute_policy()
        self.step = 0

        self.mean = torch.zeros((1, self.posterior.x_dim), device=self.device)
        self.cov = torch.eye(self.posterior.x_dim, device=self.device).unsqueeze(0)

    def __call__(self, y, u):

        """
            inputs: y_t, u_{t-1}
            outputs: planned u_t
        """

        # convert y_t to a torch tensor and add a batch dimension
        y = torch.as_tensor(y, device=self.device).unsqueeze(0)

        # no learning takes place here
        with torch.no_grad():
            self.encoder.eval()
            self.posterior.eval()
        
            a = self.encoder(y)

            # update belief using u_{t-1}
            self.mean, self.cov = self.posterior.dynamics_update(
                mean=self.mean,
                cov=self.cov,
                u=torch.as_tensor(u, device=self.device).unsqueeze(0)
            )

            # update belief using y_t
            self.mean, self.cov = self.posterior.measurement_update(
                mean=self.mean,
                cov=self.cov,
                a=a,
            )

            planned_u = self.mean @ self.Ks[self.step].T + self.ks[self.step].T
        
        self.step += 1
        return np.clip(planned_u.cpu().numpy(), a_min=-1.0, a_max=1.0)
    
    def _compute_policy(self):
        x_dim, u_dim = self.posterior.B.shape

        Ks = []
        ks = []

        V = torch.zeros((x_dim, x_dim), device=self.device)
        v = torch.zeros((x_dim, 1), device=self.device)

        C = torch.block_diag(self.cost_model.Q, self.cost_model.R)
        c = torch.cat([
            self.cost_model.q,
            torch.zeros((u_dim, 1), device=self.device)
        ])

        F = torch.cat((self.posterior.A, self.posterior.B), dim=1)
        f = torch.zeros((x_dim, 1), device=self.device)

        for _ in range(self.planning_horizon-1, -1, -1):
            Q = C + F.T @ V @ F
            q = c + F.T @ V @ f + F.T @ v
            Qxx = Q[:x_dim, :x_dim]
            Qxu = Q[:x_dim, x_dim:]
            Qux = Q[x_dim:, :x_dim]
            Quu = Q[x_dim:, x_dim:]
            qx = q[:x_dim, :]
            qu = q[x_dim:, :]

            K = - torch.linalg.pinv(Quu) @ Qux
            k = - torch.linalg.pinv(Quu) @ qu
            V = Qxx + Qxu @ K + K.T @ Qux + K.T @ Quu @ K
            v = qx + Qxu @ k + K.T @ qu + K.T @ Quu @ k

            Ks.append(K)
            ks.append(k)
        
        return Ks[::-1], ks[::-1]
    
    def reset(self):
        self.step = 0
        self.mean = torch.zeros((1, self.posterior.x_dim), device=self.device)
        self.cov = torch.eye(self.posterior.x_dim, device=self.device).unsqueeze(0)


class MPCAgent:
    """
        action planning by the LQR method
    """
    def __init__(
        self,
        encoder,
        posterior,
        cost_model,
        planning_horizon: int,
        action_noise: float = 0.3
    ):
        self.encoder = encoder
        self.posterior = posterior
        self.cost_model = cost_model
        self.planning_horizon = planning_horizon
        self.action_noise = action_noise

        self.device = next(encoder.parameters()).device

        x_dim, u_dim = self.posterior.B.shape

        C = torch.block_diag(self.cost_model.Q, self.cost_model.R).repeat(
            self.planning_horizon, 1, 1, 1,
        )

        c = torch.cat([
            self.cost_model.q.reshape(1, -1),
            torch.zeros((1, u_dim), device=self.device)
        ], dim=1).repeat(self.planning_horizon, 1, 1)

        F = torch.cat((self.posterior.A, self.posterior.B), dim=1).repeat(
            self.planning_horizon, 1, 1, 1
        )
        f = torch.zeros((1, x_dim), device=self.device).repeat(
            self.planning_horizon, 1, 1
        )

        self.quadcost = QuadCost(C, c)
        self.lindx = LinDx(F, f)

        self.planner = mpc.MPC(
            n_batch=1,
            n_state=x_dim,
            n_ctrl=u_dim,
            T=self.planning_horizon,
            u_lower=-1.0,
            u_upper=1.0,
            lqr_iter=50,
            backprop=False,
            exit_unconverged=False,
        )

        self.mean = torch.zeros((1, self.posterior.x_dim), device=self.device)
        self.cov = torch.eye(self.posterior.x_dim, device=self.device).unsqueeze(0)

    def __call__(self, y, u, explore: bool=False):

        """
            inputs: y_t, u_{t-1}
            outputs: planned u_t
        """

        # convert y_t to a torch tensor and add a batch dimension
        y = torch.as_tensor(y, device=self.device).unsqueeze(0)

        # no learning takes place here
        with torch.no_grad():
            self.encoder.eval()
            self.posterior.eval()
        
            a = self.encoder(y)

            # update belief using u_{t-1}
            self.mean, self.cov = self.posterior.dynamics_update(
                mean=self.mean,
                cov=self.cov,
                u=torch.as_tensor(u, device=self.device).unsqueeze(0)
            )

            # update belief using y_t
            self.mean, self.cov = self.posterior.measurement_update(
                mean=self.mean,
                cov=self.cov,
                a=a,
            )

            planned_x, planned_u, _ = self.planner(
                self.mean,
                self.quadcost,
                self.lindx
            )

            if explore:
                planned_u += self.action_noise * torch.randn_like(planned_u)
        
        return np.clip(planned_u.squeeze(1).cpu().numpy(), a_min=-1.0, a_max=1.0)
    
    def reset(self):
        self.mean = torch.zeros((1, self.posterior.x_dim), device=self.device)
        self.cov = torch.eye(self.posterior.x_dim, device=self.device).unsqueeze(0)