import numpy as np
import torch


class LQRAgent:
    """
        action planning by the LQR method
    """
    def __init__(
        self,
        encoder,
        posterior,
        cost_function,
        planning_horizon: int,
    ):
        self.encoder = encoder
        self.posterior = posterior
        self.cost_function = cost_function
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
        
            target = self.cost_function.target
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

            planned_u = (self.mean - target) @ self.Ks[self.step].T + self.ks[self.step].T
        
        self.step += 1
        return np.clip(planned_u.cpu().numpy(), min=-1.0, max=1.0)
    
    def _compute_policy(self):
        x_dim, u_dim = self.posterior.B.shape

        Ks = []
        ks = []

        V = torch.zeros((x_dim, x_dim), device=self.device)
        v = torch.zeros((x_dim, 1), device=self.device)

        C = torch.block_diag(self.cost_function.Q, self.cost_function.R)
        c = torch.zeros((x_dim + u_dim, 1), device=self.device)

        F = torch.cat((self.posterior.A, self.posterior.B), dim=1)
        f = (self.posterior.A - torch.eye(x_dim, device=self.device))@ self.cost_function.target.T

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