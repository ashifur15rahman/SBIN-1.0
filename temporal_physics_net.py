import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import torch
import torch.nn as nn

# ──────────────────────────────────────────────────────────────────────────────
#                             Model Definitions
# ──────────────────────────────────────────────────────────────────────────────

class TemporalPhysicsNet(nn.Module):
    """
    A 1D CNN that maps [features + time] → θ sequence,
    and two 1x1 convs for mass and potential.
    """

    def __init__(self, feat_dim=4, coord_dim=4, hidden=64, kernel=5):
        super().__init__()
        pad = kernel // 2
        # Feature → θ
        self.conv_f = nn.Sequential(
            nn.Conv1d(feat_dim+1, hidden, kernel, padding=pad),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel, padding=pad),
            nn.ReLU(),
            nn.Conv1d(hidden, coord_dim, kernel, padding=pad)
        )
        # θ + time → mass diagonal entries
        self.conv_m = nn.Sequential(
            nn.Conv1d(coord_dim+1, coord_dim, 1),
            nn.Softplus()
        )
        # θ + time → potential scalar
        self.conv_v = nn.Conv1d(coord_dim+1, 1, 1)

    def forward(self, x_feat, x_time):
        # x_feat: (B,feat_dim,T), x_time: (B,1,T)
        inp = torch.cat([x_feat, x_time], dim=1)
        theta = self.conv_f(inp)                       # (B,coord_dim,T)
        m_in = torch.cat([theta, x_time], dim=1)
        mass = self.conv_m(m_in) + 1e-3                # (B,coord_dim,T)
        v = self.conv_v(m_in).squeeze(1)               # (B,T)
        return theta, mass, v

# ──────────────────────────────────────────────────────────────────────────────
#                             Helper Functions
# ──────────────────────────────────────────────────────────────────────────────

def pad_diff(tensor, diffs, dim=2):
    """
    Given diffs = torch.diff(tensor, dim=dim), pad front/back by repeating edges.
    """
    front = diffs.narrow(dim, 0, 1)
    back  = diffs.narrow(dim, diffs.size(dim)-1, 1)
    return torch.cat([front, diffs, back], dim=dim)

# ──────────────────────────────────────────────────────────────────────────────
#                             Training Loop
# ──────────────────────────────────────────────────────────────────────────────

def train(colab_csv='20220712_0_1.csv',
          epochs=100, lr=1e-3, clusters=4, device='cuda' if torch.cuda.is_available() else 'cpu'):

    # 1) Load & preprocess
    df = pd.read_csv(colab_csv)
    df.columns = df.columns.str.strip()
    feat_np = df[['lat','lon','vn','ve']].values.astype(np.float32)  # (N,4)
    t_np    = df['time'].values.astype(np.float32)                   # (N,)

    N, D = feat_np.shape

    # 2) Build relation tensor R from KMeans
    km = KMeans(n_clusters=clusters, random_state=0).fit(feat_np)
    centers = torch.tensor(km.cluster_centers_, device=device, dtype=torch.float32)  # (C,D)
    R = centers.t() @ centers                                                        # (D,D)

    # 3) Prepare tensors for model
    feat = torch.tensor(feat_np.T[None], device=device)   # (1,4,N)
    time = torch.tensor(t_np[None,None], device=device)  # (1,1,N)

    # 4) Instantiate model & optimizer
    model = TemporalPhysicsNet(feat_dim=D, coord_dim=D, hidden=128).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history = []

    for ep in range(1, epochs+1):
        model.train()
        opt.zero_grad()

        # 5) Forward pass: get θ, M, V sequences
        theta, mass, V = model(feat, time)
        # shapes: theta (1,D,N), mass (1,D,N), V (1,N)

        # 6) Time derivatives via torch.diff (works with autograd)
        dt = time.squeeze(0).squeeze(0)[1:] - time.squeeze(0).squeeze(0)[:-1]  # (N-1,)
        theta_dot = torch.diff(theta, dim=2) / dt[None,None]                  # (1,D,N-1)
        theta_ddot= torch.diff(theta_dot, dim=2) /耗間に2 = torch.diff(theta_ddot, dim=2) / dt[None, None][:, :-1]  # (1,D,N-2)

        # 7) Pad diffs to shape N
        theta_dot_pad  = pad_diff(theta, theta_dot, dim=2)   # (1,D,N)
        theta_ddot_pad = pad_diff(theta, theta_ddot, dim=2)  # (1,D,N)

        # 8) Euler–Lagrange residual: d/dt(M θ̇) + ∂V/∂θ ≈ 0
        P    = mass * theta_dot_pad                                   # (1,D,N)
        dP   = pad_diff(P, torch.diff(P, dim=2) / dt[None,None], dim=2)   # (1,D,N)
        dVdθ = torch.autograd.grad(V.sum(), theta, create_graph=True)[0]   # (1,D,N)
        L_EL = torch.mean((dP + dVdθ).pow(2))

        # 9) Hamiltonian residual: enforce dH/dt ≈ 0
        p    = mass * theta_dot_pad
        H    = 0.5 * (p**2 / mass).sum(1) + V                           # (1,N)
        dH   = pad_diff(H, torch.diff(H, dim=1) / dt[None], dim=1)      # (1,N)
        L_HH = torch.mean(dH.pow(2))

        # 10) Brachistochrone cost
        ds   = torch.sqrt(1 + (theta_dot_pad**2).sum(1))   # (1,N)
        y    = torch.norm(theta, dim=1).clamp(min=1e-6)    # (1,N)
        L_BC = torch.trapz(ds / torch.sqrt(2*y), time.squeeze(0))  # scalar

        # 11) Relation tensor constraint
        # θᵀ R θ for each time, then mean square
        θ = theta.squeeze(0).T                            # (N,D)
        proj = torch.einsum('nd,dk,nk->n', θ, R, θ)        # (N,)
        L_RC = torch.mean(proj.pow(2))

        # 12) Total loss & backward
        loss = L_EL + L_HH + L_BC + 0.1 * L_RC
        loss.backward()
        opt.step()

        loss_history.append(loss.item())
        if ep % 10 == 0 or ep == 1:
            print(f"Epoch {ep:3d}/{epochs}  Loss: {loss.item():.3e}")

    # 13) Plot convergence
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title('SBIN Physics‑ML Training')
    plt.show()

if __name__ == "__main__":
    train()