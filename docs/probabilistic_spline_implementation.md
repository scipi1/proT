# Implementation: Replace RNN with Probabilistic Spline Regression

**Objective:** Predict a smooth, monotonically increasing time series, potentially containing a sharp terminal failure spike, from a fixed context vector $\mathbf{x}$. Replace the RNN (whose sequential inputs are only positions) with an MLP and a time-dependent basis expansion.

## 1. Simple model: MLP + Gaussian B-splines

### Architecture

1. Normalize time to $t\in[0,1]$.
2. Construct $K$ cubic B-spline basis functions $\phi_k(t)$. Start with $K=16$–$32$, using denser knots near the end if failures concentrate there.
3. Replace the RNN with an MLP taking the original context vector $\mathbf{x}$.
4. Output $2K$ parameters: coefficient means $\boldsymbol\mu_a$ and log-variances $\log\boldsymbol\sigma_a^2$.
5. Reconstruct the mean and variance of the entire time series analytically.

### Equations

Assume conditionally independent Gaussian coefficients:

$$
a_k\mid\mathbf x\sim\mathcal N(\mu_{a,k},\sigma_{a,k}^2).
$$

For the basis matrix $\Phi\in\mathbb R^{T\times K}$,

$$
\boxed{\begin{aligned}
\boldsymbol\mu_Y &= \Phi\boldsymbol\mu_a,\\
\boldsymbol\sigma_Y^2 &= (\Phi\odot\Phi)\boldsymbol\sigma_a^2.
\end{aligned}}
$$

### Training

Use Gaussian negative log likelihood (NLL) directly on the observed cumulative values:

$$
\mathcal L_{\mathrm{NLL}}
=\frac{1}{2T}\sum_{t=1}^{T}
\left[\log \sigma_Y^2(t)+\frac{(Y_t-\mu_Y(t))^2}{\sigma_Y^2(t)}\right].
$$

Add a small variance floor for numerical stability. This is a **marginal NLL**, not the full joint trajectory likelihood.

**Monotonicity:** Standard B-splines do not guarantee it. For a strict guarantee on the **mean**, use a monotone I-spline basis and nonnegative coefficients (e.g., via `softplus`). A Gaussian distribution over unconstrained coefficients does not guarantee monotone sampled trajectories; enforcing nonnegative random coefficients changes the distribution and generally loses the closed-form Gaussian output.

## 2. Extended model: Explicit failure prediction

Separate the smooth baseline from the terminal spike:

$$
\boxed{Y(t)=B(t)+Z\,A\,h(t;\tau,w)}
$$

where:

- $B(t)$: baseline predicted by the spline model.
- $Z\sim\operatorname{Bernoulli}(p)$: failure occurrence.
- $A>0$: spike amplitude.
- $\tau\in[0,1]$: failure onset.
- $w>0$: spike width.
- $h(t;\tau,w)=\operatorname{sigmoid}((t-\tau)/w)$: smooth transition.

### Architecture

Use a shared MLP with separate output heads:

| Head | Outputs | Activation |
|---|---|---|
| Baseline | $K$ coefficient means + $K$ variances | Linear / positive variance |
| Failure occurrence | $p$ (or its logit) | Sigmoid for probability |
| Spike amplitude | $A$ | Softplus |
| Spike onset | $\tau$ | Sigmoid |
| Spike width | $w$ | Softplus |

### Training

If failure labels and spike parameters are available, use

$$
\mathcal L=\mathcal L_{\mathrm{baseline}}+\lambda_f\mathcal L_{\mathrm{BCE}}+\lambda_s\mathcal L_{\mathrm{spike}}.
$$

Apply spike-parameter supervision only to failed samples. The baseline target should exclude the spike contribution (estimated through decomposition), or use a reconstruction objective on the full curve.

If failure labels or parameters are unavailable, train with a two-component **marginal** mixture likelihood:

$$
p(Y_t\mid\mathbf x)=(1-p)\,\mathcal N(Y_t;\mu_B(t),v_B(t))
+p\,\mathcal N(Y_t;\mu_B(t)+Ah(t;\tau,w),v_B(t)).
$$

Compute its NLL using `logsumexp`. If the failure event is shared by the entire trajectory, a **trajectory-level mixture likelihood** is more coherent than mixing independently at each time point.

## 3. Minimal PyTorch structure

```python
import torch
from torch import nn
from torch.nn import functional as F

class SplinePredictor(nn.Module):
    def __init__(self, context_dim, hidden_dim, K, failure=False):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.coefficients = nn.Linear(hidden_dim, 2 * K)
        self.failure = failure
        if failure:
            self.failure_head = nn.Linear(hidden_dim, 4)

    def forward(self, context, Phi):
        # context: (batch, context_dim); Phi: (T, K)
        h = self.encoder(context)

        mean_a, logvar_a = self.coefficients(h).chunk(2, dim=-1)
        var_a = logvar_a.clamp(-12, 8).exp()

        mean_y = mean_a @ Phi.T
        var_y = var_a @ Phi.square().T + 1e-6

        if not self.failure:
            return mean_y, var_y

        logits, raw_A, raw_tau, raw_w = self.failure_head(h).unbind(-1)
        A = F.softplus(raw_A)
        tau = torch.sigmoid(raw_tau)
        w = F.softplus(raw_w) + 1e-3

        # Reconstruct spike using separately stored normalized time grid t:
        # spike = A[:, None] * torch.sigmoid((t[None, :] - tau[:, None]) / w[:, None])
        # Return failure parameters for mixture/reconstruction losses.
        return mean_y, var_y, logits, A, tau, w
```

The code assumes a **precomputed spline basis** `Phi` and a separately stored normalized time grid `t` for the optional failure term. It is an architectural skeleton, not a complete loss/training script.

## 4. Recommended implementation order

1. Implement the simple Gaussian B-spline model first, keeping the same context representation.
2. Evaluate cumulative NLL, time-dependent interval coverage, monotonicity violations, and spike prediction error.
3. If rare terminal spikes remain poorly represented, add the explicit failure head instead of indiscriminately increasing spline complexity.

**Computational advantage:** The entire time series is reconstructed through matrix multiplication $\Phi\mathbf a$; there is no recurrent processing across output time steps.
