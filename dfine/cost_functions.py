import torch


class Quadratic:

    """
        c(x, u) = 0.5 * (x-x*).T @ Q @ (x-x*) + 0.5 * a.T @ R @ a
    """

    def __init__(self, Q, R, target, device: str="cpu"):
        """
            Q: x x
            R: u u
            x_target: 1 x
        """

        self.device = device
        self.Q = torch.as_tensor(Q, device=self.device, dtype=torch.float32)
        self.R = torch.as_tensor(R, device=self.device, dtype=torch.float32)
        self.target = torch.as_tensor(target, device=self.device, dtype=torch.float32)

    
    def __call__(self, x, ):
        """
            state: b x
            action: b u
        """
        x = torch.as_tensor(x, device=self.device, dtype=torch.float32)
        u = torch.as_tensor(u, device=self.device, dtype=torch.float32)

        cost = 0.5 * (x - self.target) @ self.Q @ (x - self.target).T + 0.5 * u @ self.R @ u.T
        return cost.diag()