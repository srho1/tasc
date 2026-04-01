from genData.cricket import DataCollection 
import numpy as np 
import pandas as pd 
import torch 
from torch import nn
import matplotlib.pyplot as plt 

from syslibutils import *

from tasc import TimeAwareSC 
from matrix import Matrix 
from synthetic_control import SyntheticControl 

import random 
from sklearn.metrics import root_mean_squared_error
import csv

NUM_REPS = 100 

def rsc_prediction(Y, T0, d, rscmethod='ridge', rsclmbda=0.1):
    Y = Y.detach().numpy()
    df = pd.DataFrame(Y)
    M = Matrix(df, T0, target_name=0)
    M.denoise(num_sv=d)
    sys = SyntheticControl()
    sys.fit(M.pre_donor, M.pre_target, method=rscmethod, lmbda=rsclmbda)
    pred_rsc = np.array(sys.predict(M.donor)).squeeze()
    return pred_rsc

def synthetic_control_prediction(Y, T0):
    Y = Y.detach().numpy()
    T, N = Y.shape
    Y_mask = Y.copy()
    Y_mask[T0:, 0] = [None] * (T - T0)
    df = pd.DataFrame(Y_mask)
    M = Matrix(df, T0, target_name=0)
    sys = SyntheticControl()
    sys.fit(M.pre_donor, M.pre_target, method='simplex')
    pred_sc = np.array(sys.predict(M.donor)).squeeze()
    return pred_sc


def kalman_prediction(Y, T0, N1, d, seed):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    T, N = Y.shape
    Y_mask = Y.clone()
    Y_mask[T0:, 0] = float('nan') 
    model = TimeAwareSC(Y=Y_mask.T.to(device), d=d, device=device, dtype=torch.float32)
    model.initialize_theta(method='naive', random_seed=seed)
    model.T0 = T0
    model.em_pre(T0=T0, N1=N1)
    log_like = model.log_likelihood(T=T0).item()
    with torch.no_grad():
        target_pred, donor_pred, tasc_target_var_estimates = model.make_prediction()
        R_tasc = np.diag(model.R.cpu().numpy())
    
    return log_like, target_pred.detach().numpy(), tasc_target_var_estimates, R_tasc

def cim_prediction(Y, T0):
    Y_mask = Y.clone()
    T, N = Y.shape
    Y_mask[T0:, 0] = float('nan') 
    df = pd.DataFrame(Y_mask, columns=["y"]+[f"x{i}" for i in range(N-1)])
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

def get_cricket_data():
    dc = DataCollection(True, 1, folder_path = '../../data/cricket/ipl_jsonv2')
    year_data, year_data_wicket = dc.generate_data()
    i = 120 # limit at 120
    time_slice = year_data.iloc[:, i]
    nan_rows = np.where(time_slice.isna())[0]
    year_data = year_data[~year_data.index.isin(nan_rows)]

    year_data = year_data.sort_values(by=["date"]).reset_index(drop=True)

    return year_data 

def make_placebo_Y(n, year_data):
    # From all possible games, choose random target and then return Y matrix with that target and n donors
    score_donors = torch.tensor(year_data.iloc[:, :120].values, dtype=torch.float).T
    score_targets = torch.tensor(year_data.iloc[:, :120].values,  dtype=torch.float)
    N, T = score_donors.shape
    
    target_id = np.random.randint(n+1, score_targets.shape[0])

    donor_indices = torch.arange(target_id - n, target_id)
    Y_donor = score_donors[:, donor_indices]

    Y = torch.hstack((score_targets[target_id].view(-1, 1), Y_donor))
    return Y


def placebo_test_tasc(result_log_name, d, n):
    TEST_FILE = 'resultLogscricket/' + result_log_name + '.csv'

    header = ["seed", "T", "T0", "N1", "n", "d", "loglikelihood",
              "trueTarget", "pred_tasc",   
              "tasc_target_var_estimates", "R_tasc"]

    with open(TEST_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    T = 120
    T0 = 72
    N1 = 1000
    year_data = get_cricket_data()
    num_reps = NUM_REPS 
    seed_error = 1000
    for i in range(num_reps):
        print(f"Iteration {i+1} / {num_reps}")
        seed = i
        set_seed(seed)
        Y = make_placebo_Y(n, year_data)
        mean_time_series = Y[:, 1:].mean(dim=1)
        Y_centered = Y - mean_time_series.view(-1, 1)

        max_retries = 10 
        attempt = 0
        success = False
        while attempt < max_retries and not success:
            try:
                logp, tasc_pred, tasc_target_var_estimates, R_tasc = kalman_prediction(
                    Y_centered, T0, N1, d, seed
                )
                success = True  

            except Exception as e:
                print(f"Attempt {attempt+1} failed with tasc init seed {seed}. Error: {e}, reinitializing tasc...")
                seed = seed_error + seed   
                attempt += 1

        if not success:
            raise RuntimeError("kalman_prediction failed after multiple retries")

        tasc_pred = tasc_pred + mean_time_series.detach().numpy()

        target_true = Y[:, 0].detach().numpy()

        identifiers = [seed, T, T0, N1, n, d]
        results = identifiers + [logp, target_true.tolist(), 
                                 tasc_pred.tolist(), 
                                 tasc_target_var_estimates.tolist(), 
                                 R_tasc.tolist()]
        
        try:
            with open(TEST_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(results)
        except Exception as e:
            print("Error writing CSV:", e)


                
def placebo_test_cim(result_log_name, d, n):
    TEST_FILE = 'resultLogscricket/' + result_log_name + '.csv'

    header = ["seed", "T", "T0", "N1", "n", "d", "trueTarget",
              "pred_cim", "cim_posterior_lower", "cim_posterior_upper"]

    with open(TEST_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    T = 120
    T0 = 72
    N1 = 1000
    year_data = get_cricket_data()
    num_reps = NUM_REPS 
    for i in range(num_reps):
        print(f"Iteration {i+1} / {num_reps}")
        seed = i
        set_seed(seed)
        set_tf_seed(seed)
        Y = make_placebo_Y(n, year_data)
        mean_time_series = Y[:, 1:].mean(dim=1)
        Y_centered = Y - mean_time_series.view(-1, 1)

        impact, cim_pred, cim_posterior_lower, cim_posterior_upper = cim_prediction(Y_centered, T0)
        cim_pred = cim_pred + mean_time_series.detach().numpy()

        target_true = Y[:, 0].detach().numpy()

        identifiers = [seed, T, T0, N1, n, d]
        results = identifiers + [target_true.tolist(), 
                                 cim_pred.tolist(),
                                 cim_posterior_lower.tolist(), 
                                 cim_posterior_upper.tolist()
                                 ]
        
        try:
            with open(TEST_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(results)
        except Exception as e:
            print("Error writing CSV:", e)

def placebo_test_sc_rsc(result_log_name, d, n):
    TEST_FILE = 'resultLogscricket/' + result_log_name + '.csv'

    header = ["seed", "T", "T0", "N1", "n", "d", 
              "trueTarget", "pred_rsc", "pred_sc"]

    with open(TEST_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    T = 120
    T0 = 72
    N1 = 1000
    year_data = get_cricket_data()
    num_reps = NUM_REPS 
    for i in range(num_reps):
        print(f"Iteration {i+1} / {num_reps}")
        seed = i
        set_seed(seed)
        Y = make_placebo_Y(n, year_data)
        mean_time_series = Y[:, 1:].mean(dim=1)
        Y_centered = Y - mean_time_series.view(-1, 1)

        rsc_ridge_pred = rsc_prediction(Y_centered, T0, d, rscmethod='ridge', rsclmbda=1000.0)
        rsc_ridge_pred = rsc_ridge_pred + mean_time_series.detach().numpy()

        sc_pred =  synthetic_control_prediction(Y_centered, T0)
        sc_pred = sc_pred + mean_time_series.detach().numpy()

        target_true = Y[:, 0].detach().numpy()

        identifiers = [seed, T, T0, N1, n, d]
        results = identifiers + [target_true.tolist(), 
                                 rsc_ridge_pred.tolist(), 
                                 sc_pred.tolist(), 
                                ]
        
        try:
            with open(TEST_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(results)
        except Exception as e:
            print("Error writing CSV:", e)

def rsc_params_test(result_log_name, d, n):
    TEST_FILE = 'resultLogscricket/' + result_log_name + '.csv'

    header = ["seed", "T", "T0", "N1", "n", "d", 
              "trueTarget", "pred_rsc_0_1", "pred_rsc_1_0", "pred_rsc_10_0", "pred_rsc_100_0", "pred_rsc_1000_0", "pred_rsc_10000_0", "pred_rsc_100000_0", "pred_rsc_1000000_0"]

    with open(TEST_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    T = 72 
    T0 = 58
    N1 = 1000
    year_data = get_cricket_data()
    num_reps = 100 
    for i in range(num_reps):
        print(f"Iteration {i+1} / {num_reps}")
        seed = i
        set_seed(seed)
        Y = make_placebo_Y(n, year_data)
        Y = Y[:T, :]

        mean_time_series = Y[:, 1:].mean(dim=1)
        Y_centered = Y - mean_time_series.view(-1, 1)

        rsc_pred_0_1 = rsc_prediction(Y_centered, T0, d, rscmethod="ridge", rsclmbda=0.1)
        rsc_pred_0_1 = rsc_pred_0_1 + mean_time_series.detach().numpy()

        rsc_pred_1_0 = rsc_prediction(Y_centered, T0, d, rscmethod="ridge", rsclmbda=1.0)
        rsc_pred_1_0 = rsc_pred_1_0 + mean_time_series.detach().numpy()

        rsc_pred_10_0 = rsc_prediction(Y_centered, T0, d, rscmethod="ridge", rsclmbda=10.0)
        rsc_pred_10_0 = rsc_pred_10_0 + mean_time_series.detach().numpy()

        rsc_pred_100_0 = rsc_prediction(Y_centered, T0, d, rscmethod="ridge", rsclmbda=100.0)
        rsc_pred_100_0 = rsc_pred_100_0 + mean_time_series.detach().numpy()

        rsc_pred_1000_0 = rsc_prediction(Y_centered, T0, d, rscmethod="ridge", rsclmbda=1000.0)
        rsc_pred_1000_0 = rsc_pred_1000_0 + mean_time_series.detach().numpy()

        rsc_pred_10000_0 = rsc_prediction(Y_centered, T0, d, rscmethod="ridge", rsclmbda=10000.0)
        rsc_pred_10000_0 = rsc_pred_10000_0 + mean_time_series.detach().numpy()

        rsc_pred_100000_0 = rsc_prediction(Y_centered, T0, d, rscmethod="ridge", rsclmbda=100000.0)
        rsc_pred_100000_0 = rsc_pred_100000_0 + mean_time_series.detach().numpy()

        rsc_pred_1000000_0 = rsc_prediction(Y_centered, T0, d, rscmethod="ridge", rsclmbda=1000000.0)
        rsc_pred_1000000_0 = rsc_pred_1000000_0 + mean_time_series.detach().numpy()

        target_true = Y[:, 0].detach().numpy()

        identifiers = [seed, T, T0, N1, n, d]
        results = identifiers + [target_true.tolist(), 
                                 rsc_pred_0_1.tolist(),
                                 rsc_pred_1_0.tolist(),
                                 rsc_pred_10_0.tolist(), 
                                 rsc_pred_100_0.tolist(), 
                                 rsc_pred_1000_0.tolist(), 
                                 rsc_pred_10000_0.tolist(), 
                                 rsc_pred_100000_0.tolist(), 
                                 rsc_pred_1000000_0.tolist()
                                 ]
        
        try:
            with open(TEST_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(results)
        except Exception as e:
            print("Error writing CSV:", e)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False

def set_tf_seed(seed: int = 42):
    tf.random.set_seed(seed)

if __name__ == "__main__":
    rsc_params_test("cricket_placebo_coeff_rsc_n72_d5_T058_aistats", 5, 72)

    placebo_test_sc_rsc("cricket_placebo_test_rsc_sc_n18_d5_T072_aistats", 5, 18)
    placebo_test_sc_rsc("cricket_placebo_test_rsc_sc_n36_d5_T072_aistats", 5, 36)
    placebo_test_sc_rsc("cricket_placebo_test_rsc_sc_n72_d5_T072_aistats", 5, 72)
    placebo_test_sc_rsc("cricket_placebo_test_rsc_sc_n144_d5_T072_aistats", 5, 144)
    placebo_test_sc_rsc("cricket_placebo_test_rsc_sc_n288_d5_T072_aistats", 5, 288)

    placebo_test_tasc("cricket_placebo_test_tasc_n18_d5_T072_aistats", 5, 18)
    placebo_test_tasc("cricket_placebo_test_tasc_n36_d5_T072_aistats", 5, 36)
    placebo_test_tasc("cricket_placebo_test_tasc_n72_d5_T072_aistats", 5, 72)
    placebo_test_tasc("cricket_placebo_test_tasc_n144_d5_T072_aistats", 5, 144)
    placebo_test_tasc("cricket_placebo_test_tasc_n288_d5_T072_aistats", 5, 288)


    ## Now working with tensorflow so use the virtual environment
    ## It is best to run the below tests seperately in an environment with tensorflow install. 
    ## We found on some machines importing tensorflow and causalimpact impacted classic SC performance
    ## TASC and RSC are not impacted by causalimpact and tensorflow library included 

    # import tensorflow as tf 
    # import tensorflow_probability as tfp
    # import causalimpact

    # placebo_test_cim("cricket_placebo_test_cim_n18_d5_T072_aistats", 5, 18)
    # placebo_test_cim("cricket_placebo_test_cim_n36_d5_T072_aistats", 5, 36)
    # placebo_test_cim("cricket_placebo_test_cim_n72_d5_T072_aistats", 5, 72)
    # placebo_test_cim("cricket_placebo_test_cim_n144_d5_T072_aistats", 5, 144)
    # placebo_test_cim("cricket_placebo_test_cim_n288_d5_T072_aistats", 5, 288)