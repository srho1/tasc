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
from tasc import TimeAwareSC as TASC

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

result_log_name = "exp0-test"
# TEST_FILE = 'logs/' + result_log_name + '.csv'
TEST_FILE = 'KalmanSC/test/aistats_exp1/logs/' + result_log_name + '.csv'

header = ["seed_datagen", "seed_learning", "d_true", "N", "T", "T0", "N1", "d", "high_covariance",
        "loglikelihood", "loglikelihood_permute",
        "pred_tasc", "tasc_target_var_estimates", 
        "pred_tasc_permute", "tasc_permute_target_var_estimates",
        "pred_cim", "cim_posterior_lower", "cim_posterior_upper",
        "pred_cim_permute", "cim_permute_posterior_lower", "cim_permute_posterior_upper",
        "R_tasc", "R_tasc_permute",
        "ys", "ys_signal",
        ]

with open(TEST_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)

for seed_datagen in range(100):
    set_seed(seed_datagen)

    # #### for real
    # d_true = 5
    # d = 5
    # N, T = 50, 100
    # T0 = 50

    #### for test
    d_true = 2
    d = 2
    N, T = 3, 5
    T0 = 4

    # for variance noise
    low = 0.1
    high = 1
    high_covariance = "both"

    
    # generate data
    theta_true = gen_dirchelet_params(d=d_true, N=N, noise_min_q=low, noise_max_q=high, noise_min_r=low, noise_max_r=high,Q_diag=False, R_diag=False, random_seed=seed_datagen) #[A, H, Q, R, m0, P0]
    ys_signal, ys = generate_model_data(theta_true, T=T, return_signal=True, random_seed=seed_datagen) # T by N

    # permute data
    Y = torch.tensor(ys.T, dtype=torch.float32)    # (N, T)
    Y_permute= Y.clone()    # (N, T)
    # prepare pre
    Y_pre = Y_permute[:,:T0]
    shuffler_pre = IndexShuffler(n=T0)
    Y_pre = Y_pre[:,shuffler_pre.permutation]
    # prepare post
    Y_post = Y_permute[:,T0:]
    shuffler_post = IndexShuffler(n=T-T0)
    Y_post = Y_post[:,shuffler_post.permutation]
    Y_permute = torch.cat([Y_pre, Y_post], dim=1)
    # mask
    Y_permute[0,T0:] = 0    # mask
    ys_permute = Y_permute.T.numpy()  # (T, N)

    # CIM baseline
    impact, cim_pred, cim_posterior_lower, cim_posterior_upper = cim_prediction(ys, T, T0)
                
    # CIM permute
    impact_permute, cim_permute_pred, cim_permute_posterior_lower, cim_permute_posterior_upper = cim_prediction(ys_permute, T, T0)
    cim_permute_pred[T0:] = cim_permute_pred[T0:][shuffler_post.inverse]
    cim_permute_posterior_lower[T0:] = cim_permute_posterior_lower[T0:][shuffler_post.inverse]
    cim_permute_posterior_upper[T0:] = cim_permute_posterior_upper[T0:][shuffler_post.inverse]

    seed_learning = None
    N1=None

    # TASC
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
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

            model = TASC(Y=Y_mask.to(device), d=d, device=device, dtype=torch.float32)
            model.initialize_theta(method='dirichlet', random_seed=seed_learning)
            model.T0 = T0

            # EM
            model.em_pre(T0=T0, N1=N1)
            logp = model.log_likelihood(T=T0).item()
            with torch.no_grad():
                tasc_pred, donor_pred, tasc_target_var_estimates = model.make_prediction()
                R_tasc = np.diag(model.R.cpu().numpy())

            ######################
            ## PERMUTE TASC

            model = TASC(Y=Y_permute.to(device), d=d, device=device, dtype=torch.float32)
            model.initialize_theta(method='dirichlet', random_seed=seed_learning)
            model.T0 = T0

            # EM
            model.em_pre(T0=T0, N1=N1)
            logp_permute = model.log_likelihood(T=T0).item()
            with torch.no_grad():
                tasc_permute_pred, donor_pred, tasc_permute_target_var_estimates = model.make_prediction()
                tasc_permute_pred[T0:] = tasc_permute_pred[T0:][shuffler_post.inverse]
                R_tasc_permute = np.diag(model.R.cpu().numpy())

            ######################
            # save results
            identifiers = [seed_datagen, seed_learning, d_true, N, T, T0, N1, d, high_covariance]
            results = identifiers + [logp, logp_permute,
                                        # OG TASC
                                        tasc_pred.tolist(),
                                        tasc_target_var_estimates.tolist(),
                                        # PERMUTE TASC
                                        tasc_permute_pred.tolist(),
                                        tasc_permute_target_var_estimates.tolist(),
                                        # OG CIM
                                        cim_pred.tolist(),
                                        cim_posterior_lower.tolist(),
                                        cim_posterior_upper.tolist(),
                                        # PERMUTE CIM
                                        cim_permute_pred.tolist(),
                                        cim_permute_posterior_lower.tolist(),
                                        cim_permute_posterior_upper.tolist(),
                                        # variances
                                        R_tasc.tolist(),
                                        R_tasc_permute.tolist(),
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
    print(seed_datagen)
    print("--------------------------")
