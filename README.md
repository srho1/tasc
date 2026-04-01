# Time-Aware Synthetic Control

This repository contains the code for the following paper, published in AISTATS 2026.

**Paper:** [Time-Aware Synthetic Control](https://arxiv.org/abs/2601.03099)

## Installation

```bash
pip install -e .
```

## Usage

`TimeAwareSC` estimates a counterfactual for the treated unit (row 0 of `Y`) using the donor units (rows 1..N-1) via a Kalman filter / EM algorithm.

### Input format

`Y` is an `(N, T)` matrix where:
- `N` = number of units (row 0 is the treated unit, rows 1..N-1 are donors)
- `T` = total number of time periods
- `T0` = last pre-intervention time period (intervention occurs at `T0+1`)

### Basic example

```python
import torch
from tasc import TimeAwareSC

# Y: (N, T) panel — row 0 is the treated unit
Y = torch.tensor(your_panel_data, dtype=torch.float32)  # shape (N, T)

N, T = Y.shape
T0 = 50   # pre-intervention periods
d  = 5    # latent state dimension (hyperparameter)

model = TimeAwareSC(Y=Y, d=d, T0=T0)

# Initialize parameters (options: 'pca', 'dirichlet', 'naive')
model.initialize_theta(method='pca')

# Fit with EM: N1 pre-intervention iterations, N2 post-intervention iterations
model.em_full(T0=T0, N1=200, N2=50)

# Predict the counterfactual for the treated unit over all T periods
with torch.no_grad():
    target_pred, donor_pred, target_var = model.make_prediction()

# target_pred: (T,)  — counterfactual trajectory for the treated unit
# target_var:  (T,)  — variance estimates for the prediction
counterfactual = target_pred[T0:]   # post-intervention counterfactual
```

### Step-by-step fitting

You can also run the pre- and post-intervention EM steps separately:

```python
model.initialize_theta(method='pca')

# Fit on pre-intervention data only
model.em_pre(T0=T0, N1=200)

# Extend fit to post-intervention data
model.em_post(N2=50)
```

### GPU support

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = TimeAwareSC(Y=Y.to(device), d=d, T0=T0, device=device)
```

### Key parameters

| Parameter | Description |
|-----------|-------------|
| `Y` | `(N, T)` panel matrix; row 0 is the treated unit |
| `d` | Latent state dimension |
| `T0` | Last pre-intervention time index |
| `Q_diag` / `R_diag` | Restrict process / observation noise to diagonal (default: `True`) |
| `learn_Q` / `learn_R` | Whether to learn noise covariances in EM (default: `True`) |

