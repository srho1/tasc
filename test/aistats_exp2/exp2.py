import random
import csv
import torch
from torch import nn
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, root_mean_squared_error

# generate data
from genData.SSM import *

# SC, RSC
from matrix import Matrix 
from synthetic_control import SyntheticControl

# TASC
<<<<<<< HEAD
from tasc import TimeAwareSC
=======
from tasc import TimeAwareSC 
>>>>>>> cyrus-final

# CIM
import tensorflow as tf
import tensorflow_probability as tfp
import causalimpact


def set_seed(seed: int = 42):
    random.seed(seed)              # Python random
    # np.random.seed(seed)           # NumPy (if used anywhere)
    torch.manual_seed(seed)        # Torch CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # All CUDA devices

    # For full reproducibility (slows down training a bit)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def synthetic_control_prediction(ys, T, T0):
    ys_mask = ys.copy()
    ys_mask[T0:, 0] = [None] * (T - T0)
    df = pd.DataFrame(ys_mask)
    M = Matrix(df, T0, target_name=0)
    sys = SyntheticControl()
    sys.fit(M.pre_donor, M.pre_target, method='simplex')
    sc_pred = np.array(sys.predict(M.donor)).squeeze()
    return sc_pred

def rsc_prediction(ys, T, T0, d, rscmethod='ridge', rsclmbda=0.1, bad_states=None):
    ys_mask = ys.copy()
    ys_mask[T0:, 0] = [None] * (T - T0)
    df = pd.DataFrame(ys_mask)
    M = Matrix(df, T0, target_name=0)
    M.denoise(num_sv=d)
    sys = SyntheticControl()
    sys.fit(M.pre_donor, M.pre_target, method=rscmethod, lmbda=rsclmbda)
    rsc_pred = np.array(sys.predict(M.donor)).squeeze()
    return rsc_pred

def cim_prediction(ys, T, T0):
    ys_mask = ys.copy()
    ys_mask[T0:, 0] = [None] * (T - T0)  # You can skip this to use causalimpact.plot(impact)
    df = pd.DataFrame(ys_mask, columns=["y"]+[f"x{i}" for i in range(ys_mask.shape[1]-1)])
    pre_period = (0, T0-1)
    post_period = (T0, T-1)
    impact = causalimpact.fit_causalimpact(
        data=df,
        pre_period=pre_period,
        post_period=post_period)
    cim_pred = np.array(impact.series['posterior_mean']).squeeze()
    posterior_lower = impact.series["posterior_lower"]
    posterior_upper = impact.series["posterior_upper"]
    return impact, cim_pred, posterior_lower, posterior_upper

class IndexShuffler:
    def __init__(self, n, seed=None):
        """
        Create a shuffler for indices [0, 1, ..., n-1].
        """
        self.n = n
        rng = np.random.default_rng(seed)
        self.permutation = rng.permutation(n)
        # Inverse permutation to go back
        self.inverse = np.argsort(self.permutation)

result_log_name = "exp2_try1"
TEST_FILE = 'logs/' + result_log_name + '.csv'

header = ["seed_datagen", "seed_learning", "d_true", "N", "T", "T0", "N1", "d",
        "high_covariance", "loglikelihood", "loglikelihood_permute",
        "pred_tasc", "pred_tasc_permute", "pred_rsc", "pred_sc", "pred_cim", "cim_posterior_lower", "cim_posterior_upper",
        "tasc_target_var_estimates", "tasc_permute_target_var_estimates",
        "R_tasc", "R_tasc_permute",
        "ys", "ys_signal",
        ]

with open(TEST_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)

for seed_datagen in range(100):
    set_seed(seed_datagen)

    #### for real
    d_true = 5
    N, T = 50, 100
    T0 = 50

    # #### for test
    # d_true = 2
    # d = 2
    # N, T = 3, 5
    # T0 = 4

    # for variance noise
    low = 0.1
    high = 1

    for d in [3, 10, 20]:

        # data has moderate variance in both Q and R
        high_covariance = "both"
        # theta_true = gen_dirchelet_params(d=d_true, N=N, noise_min_r=low, noise_max_r=high, Q_diag=False, R_diag=False, random_seed=seed_datagen) #[A, H, Q, R, m0, P0]
        theta_true = gen_dirchelet_params(d=d_true, N=N, noise_min_q=low, noise_max_q=high, noise_min_r=low, noise_max_r=high,Q_diag=False, R_diag=False, random_seed=seed_datagen) #[A, H, Q, R, m0, P0]

        ys_signal, ys = generate_model_data(theta_true, T=T, return_signal=True, random_seed=seed_datagen) # T by N

        # RSC baseline
        rsc_pred = rsc_prediction(ys, T, T0, d, rscmethod='ridge', rsclmbda=0.1)

        # SC baseline
        # sc_pred = synthetic_control_prediction(ys, T, T0)
        sc_pred = None

        # CIM baseline
        # impact, cim_pred, cim_posterior_lower, cim_posterior_upper = cim_prediction(ys, T, T0)
        impact, cim_pred, cim_posterior_lower, cim_posterior_upper = None, None, None, None

        # TASC
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # device = "cpu"
        # for seed_learning in range(1000, 1005, 1):
        for seed_learning in [1000]:
            print(f"Learning seed: {seed_learning}")
            try:
                set_seed(seed=seed_learning)
                N1 = 1000
                
                Y = torch.tensor(ys.T, dtype=torch.float32)    # (N, T)

                ######################
                ## OG TASC
                Y_mask = Y.clone()  # (N, T)
                Y_mask[0,T0:] = 0

                model = TimeAwareSC(Y=Y_mask.to(device), d=d, device=device, dtype=torch.float32)
                model.initialize_theta(method='dirichlet', random_seed=seed_learning)
                model.T0 = T0

                # EM
                model.em_pre(T0=T0, N1=N1)
                logp = model.log_likelihood(T=T0).item()
                with torch.no_grad():
                    tasc_pred, donor_pred, tasc_target_var_estimates = model.make_prediction()
                    R_tasc = np.diag(model.R.cpu().numpy())

                # ######################
                # ## PERMUTE TASC
                logp_permute = None
                tasc_permute_pred = None
                tasc_permute_target_var_estimates = None
                R_tasc_permute = None
                
                ######################
                # save results

                identifiers = [seed_datagen, seed_learning, d_true, N, T, T0, N1, d, high_covariance]
                results = identifiers + [logp, logp_permute,
                                            tasc_pred.tolist(),
                                            tasc_permute_pred,
                                            rsc_pred.tolist(),
                                            sc_pred,
                                            cim_pred,
                                            cim_posterior_lower,
                                            cim_posterior_upper,
                                            tasc_target_var_estimates.tolist(),
                                            tasc_permute_target_var_estimates,
                                            R_tasc.tolist(),
                                            R_tasc_permute, 
                                            ys[:,0].tolist(),
                                            ys_signal[:,0].tolist(),
                                        ]

                try:
                    with open(TEST_FILE, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(results)
                except Exception as e:
                    print("Error writing CSV:", e)


            except Exception as e:
                print("Error in training loop:", e)
                continue



    print("--------------------------")
    print("hello")
    print("--------------------------")
