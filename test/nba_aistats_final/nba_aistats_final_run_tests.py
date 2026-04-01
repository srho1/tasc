from genData.nba import NBADataCollection 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import torch 
from torch import nn 

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


def get_nba_data():
    sampling_rate = 15
    start_date="2020-01-01"
    filepath = "../../data/nba/"

    nba = NBADataCollection(sampling_rate=sampling_rate, start_date=start_date, filepath=filepath)

    ## Comment out the lines below after get_full_data the first time
    data = nba.get_full_data()
    data.to_csv(f"{filepath}nba_full_data_{sampling_rate}.csv", index=False)
    # data = pd.read_csv(f"{filepath}nba_full_data_{sampling_rate}.csv") 

    total_seconds = int(60*12*4)
    total_timepoints = int((60/sampling_rate)*12*4)

    Y_full = data.iloc[:,1:1+total_timepoints]
    Y_full = Y_full.astype(float) 

    YearData = data["game_date"]
    YearData_np = YearData.to_numpy() 
    Y_full_mean = Y_full.mean(axis=0)
    Y_full_centered = Y_full - Y_full_mean
    Y_full_np = Y_full_centered.to_numpy()

    threshold_min = -50
    threshold_max = 50

    mask = np.all((Y_full_np >= threshold_min) & (Y_full_np <= threshold_max), axis=1)
    YearData_np = YearData_np[mask]
    Y_full_np = Y_full_np[mask]
    Y_full_np = Y_full_np + Y_full_mean.to_numpy()
    Y_full = Y_full_np
    return Y_full, YearData_np

def make_placebo_Y(Y_full, YearData, n):
    # From all possible games, choose random target and then return Y matrix with that target and n donors
    Y_full = np.asarray(Y_full)
    YearData = np.asarray(YearData)

    if Y_full.ndim != 2:
        raise ValueError("Y must be 2D (units x time).")
    if YearData.shape[0] != Y_full.shape[0]:
        raise ValueError("YearData must have the same number of rows as Y.")
    if not (0 <= n < Y_full.shape[0]):
        raise ValueError("n must be >= 0 and less than the number of rows in Y.")

    total_units = Y_full.shape[0]

    target_idx = np.random.randint(0, total_units)
    target_date = YearData[target_idx]

    previous_indices = np.where(YearData < target_date)[0]

    if len(previous_indices) < n:
        raise ValueError(f"Not enough previous games ({len(previous_indices)}) before target date to select {n} donors.")

    donor_indices = previous_indices[-n:]

    Y = np.vstack((Y_full[target_idx:target_idx + 1, :], Y_full[donor_indices, :]))
    YearData_placebo = np.concatenate(([YearData[target_idx]], YearData[donor_indices]))
    return Y



def placebo_test_cim(result_log_name, d, n, T0):
    TEST_FILE = 'resultLogsnba/' + result_log_name + '.csv'

    header = ["seed", "T", "T0", "N1", "n", "d",
              "trueTarget", "pred_cim", "cim_posterior_lower", "cim_posterior_upper"]

    with open(TEST_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    T = 192 
    N1 = 1000
    game_data, year_data = get_nba_data()
    num_reps = NUM_REPS 
    seed_error = 1000
    for i in range(num_reps):
        print(f"Iteration {i+1} / {num_reps}")
        seed = i
        set_seed(seed)
        set_tf_seed(seed)
        Y = torch.from_numpy(make_placebo_Y(game_data, year_data, n).T).to(torch.float32)
        mean_time_series = Y[:, 1:].mean(dim=1)
        Y_centered = Y - mean_time_series.view(-1, 1)

        impact, cim_pred, cim_posterior_lower, cim_posterior_upper = cim_prediction(Y_centered, T0)
        cim_pred = cim_pred + mean_time_series.detach().numpy()

        target_true = Y[:, 0].detach().numpy()

        identifiers = [seed, T, T0, N1, n, d]
        results = identifiers + [target_true.tolist(), 
                                 cim_pred.tolist(),
                                 cim_posterior_lower.tolist(),
                                 cim_posterior_upper.tolist(), 
                                 ]
        
        try:
            with open(TEST_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(results)
        except Exception as e:
            print("Error writing CSV:", e)


def placebo_test_tasc(result_log_name, d, n, T0):
    TEST_FILE = 'resultLogsnba/' + result_log_name + '.csv'

    header = ["seed", "T", "T0", "N1", "n", "d", "loglikelihood",
              "trueTarget", "pred_tasc", "tasc_target_var_estimates", "R_tasc"]

    with open(TEST_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    T = 192 
    N1 = 1000
    game_data, year_data = get_nba_data()
    num_reps = NUM_REPS 
    seed_error = 1000
    for i in range(num_reps):
        print(f"Iteration {i+1} / {num_reps}")
        seed = i
        set_seed(seed)
        Y = torch.from_numpy(make_placebo_Y(game_data, year_data, n).T).to(torch.float32)
        mean_time_series = Y[:, 1:].mean(dim=1)
        Y_centered = Y - mean_time_series.view(-1, 1)

        max_retries = 50 
        attempt = 0
        success = False
        while attempt < max_retries and not success:
            try:
                # Try running with the current seed
                logp, tasc_pred, tasc_target_var_estimates, R_tasc = kalman_prediction(
                    Y_centered, T0, N1, d, seed
                )
                success = True  

            except Exception as e:
                print(f"Attempt {attempt+1} failed with seed tasc init seed {seed}. Error: {e}. Reinitializing TASC")
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


def placebo_test_rsc_sc(result_log_name, d, n, T0):
    TEST_FILE = 'resultLogsnba/' + result_log_name + '.csv'

    header = ["seed", "T", "T0", "N1", "n", "d", 
              "trueTarget", "pred_rsc", "pred_sc" ]

    with open(TEST_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    T = 192 
    N1 = 1000
    game_data, year_data = get_nba_data()
    num_reps = NUM_REPS 
    seed_error = 1000
    for i in range(num_reps):
        print(f"Iteration {i+1} / {num_reps}")
        seed = i
        set_seed(seed)
        Y = torch.from_numpy(make_placebo_Y(game_data, year_data, n).T).to(torch.float32)
        mean_time_series = Y[:, 1:].mean(dim=1)
        Y_centered = Y - mean_time_series.view(-1, 1)

        rsc_pred = rsc_prediction(Y_centered, T0, d, rscmethod='ridge', rsclmbda=1000.0)
        rsc_pred = rsc_pred + mean_time_series.detach().numpy()

        sc_pred =  synthetic_control_prediction(Y_centered, T0)
        sc_pred = sc_pred + mean_time_series.detach().numpy()

        target_true = Y[:, 0].detach().numpy()

        identifiers = [seed, T, T0, N1, n, d]
        results = identifiers + [target_true.tolist(),  
                                 rsc_pred.tolist(), 
                                 sc_pred.tolist()]
        
        try:
            with open(TEST_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(results)
        except Exception as e:
            print("Error writing CSV:", e)

def placebo_test_rsc_params(result_log_name, d, n, T0):
    TEST_FILE = 'resultLogsnba/' + result_log_name + '.csv'

    header = ["seed", "T", "T0", "N1", "n", "d", 
              "trueTarget", "pred_rsc_0_1", "pred_rsc_1", "pred_rsc_10", "pred_rsc_100", "pred_rsc_1000", "pred_rsc_10000", "pred_rsc_100000", "pred_rsc_1000000"]

    with open(TEST_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    T = 96 
    
    N1 = 1000
    game_data, year_data = get_nba_data()
    num_reps = 100 
    seed_error = 1000
    for i in range(num_reps):
        print(f"Iteration {i+1} / {num_reps}")
        seed = i 
        set_seed(seed)
        Y = torch.from_numpy(make_placebo_Y(game_data, year_data, n).T).to(torch.float32)
        Y = Y[:T, :]
        mean_time_series = Y[:, 1:].mean(dim=1)
        Y_centered = Y - mean_time_series.view(-1, 1)

        rsc_pred_0_1 = rsc_prediction(Y_centered, T0, d, rscmethod='ridge', rsclmbda=0.1)
        rsc_pred_0_1 = rsc_pred_0_1 + mean_time_series.detach().numpy()

        rsc_pred_1 = rsc_prediction(Y_centered, T0, d, rscmethod='ridge', rsclmbda=1.0)
        rsc_pred_1 = rsc_pred_1 + mean_time_series.detach().numpy()

        rsc_pred_10 = rsc_prediction(Y_centered, T0, d, rscmethod='ridge', rsclmbda=10.0)
        rsc_pred_10 = rsc_pred_10 + mean_time_series.detach().numpy()

        rsc_pred_100 = rsc_prediction(Y_centered, T0, d, rscmethod='ridge', rsclmbda=100.0)
        rsc_pred_100 = rsc_pred_100 + mean_time_series.detach().numpy()

        rsc_pred_1000 = rsc_prediction(Y_centered, T0, d, rscmethod='ridge', rsclmbda=1000.0)
        rsc_pred_1000 = rsc_pred_1000 + mean_time_series.detach().numpy()

        rsc_pred_10000 = rsc_prediction(Y_centered, T0, d, rscmethod='ridge', rsclmbda=10000.0)
        rsc_pred_10000 = rsc_pred_10000 + mean_time_series.detach().numpy()

        rsc_pred_100000 = rsc_prediction(Y_centered, T0, d, rscmethod='ridge', rsclmbda=100000.0)
        rsc_pred_100000 = rsc_pred_100000 + mean_time_series.detach().numpy()

        rsc_pred_1000000 = rsc_prediction(Y_centered, T0, d, rscmethod='ridge', rsclmbda=1000000.0)
        rsc_pred_1000000 = rsc_pred_1000000 + mean_time_series.detach().numpy()

        target_true = Y[:, 0].detach().numpy()

        identifiers = [seed, T, T0, N1, n, d]
        results = identifiers + [target_true.tolist(),  
                                 rsc_pred_0_1.tolist(), 
                                 rsc_pred_1.tolist(), 
                                 rsc_pred_10.tolist(),
                                 rsc_pred_100.tolist(),
                                 rsc_pred_1000.tolist(),
                                 rsc_pred_10000.tolist(), 
                                 rsc_pred_100000.tolist(),
                                 rsc_pred_1000000.tolist()]
        
        try:
            with open(TEST_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(results)
        except Exception as e:
            print("Error writing CSV:", e)


def placebo_test_tasc_bign(result_log_name, d, n, T0):
    TEST_FILE = 'resultLogsnba/' + result_log_name + '.csv'

    header = ["seed", "T", "T0", "N1", "n", "d", "loglikelihood",
              "trueTarget", "pred_tasc", "tasc_target_var_estimates", "R_tasc"]

    with open(TEST_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    T = 192 
    N1 = 1000
    game_data, year_data = get_nba_data()
    num_reps = NUM_REPS 
    seed_error = 1000
    for i in range(num_reps):
        print(f"Iteration {i+1} / {num_reps}")
        seed = i
        set_seed(seed)
        max_retries = 10
        seed_error = 1000
        attempt = 0
        success = False
        while attempt < max_retries and not success:
            try: 
                Y = torch.from_numpy(make_placebo_Y(game_data, year_data, n).T).to(torch.float32)
                success = True
            except Exception as e:
                print(f"Attempt {attempt + 1} failed with seed {seed}. Error: {e}")
                seed = seed_error + seed
                attempt += 1
                set_seed(seed)
        if not success:
            raise RuntimeError("getting donor failed after multiple retries")

        mean_time_series = Y[:, 1:].mean(dim=1)
        Y_centered = Y - mean_time_series.view(-1, 1)

        max_retries = 50 
        attempt = 0
        success = False
        while attempt < max_retries and not success:
            try:
                logp, tasc_pred, tasc_target_var_estimates, R_tasc = kalman_prediction(
                    Y_centered, T0, N1, d, seed
                )
                success = True  

            except Exception as e:
                print(f"Attempt {attempt+1} failed with tasc init seed {seed}. Error: {e}. Reinitializing TASC params")
                seed = seed_error + seed   
                attempt += 1

        if not success:
            raise RuntimeError("Initializing TASC failed after multiple retries")

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

def placebo_test_rsc_sc_bign(result_log_name, d, n, T0):
    TEST_FILE = 'resultLogsnba/' + result_log_name + '.csv'

    header = ["seed", "T", "T0", "N1", "n", "d", 
              "trueTarget", "pred_rsc", "pred_sc" ]

    with open(TEST_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    T = 192 
    N1 = 1000
    game_data, year_data = get_nba_data()
    num_reps = NUM_REPS 
    seed_error = 1000
    for i in range(num_reps):
        print(f"Iteration {i+1} / {num_reps}")
        seed = i
        set_seed(seed)
        max_retries = 10
        seed_error = 1000
        attempt = 0
        success = False
        while attempt < max_retries and not success:
            try: 
                Y = torch.from_numpy(make_placebo_Y(game_data, year_data, n).T).to(torch.float32)
                success = True
            except Exception as e:
                print(f"Attempt {attempt + 1} failed with seed {seed}. Error: {e}")
                seed = seed_error + seed
                attempt += 1
                set_seed(seed)
        if not success:
            raise RuntimeError("getting donor failed after multiple retries")

        mean_time_series = Y[:, 1:].mean(dim=1)
        Y_centered = Y - mean_time_series.view(-1, 1)

        rsc_pred = rsc_prediction(Y_centered, T0, d, rscmethod='ridge', rsclmbda=1000.0)
        rsc_pred = rsc_pred + mean_time_series.detach().numpy()

        sc_pred =  synthetic_control_prediction(Y_centered, T0)
        sc_pred = sc_pred + mean_time_series.detach().numpy()

        target_true = Y[:, 0].detach().numpy()

        identifiers = [seed, T, T0, N1, n, d]
        results = identifiers + [target_true.tolist(),  
                                 rsc_pred.tolist(), 
                                 sc_pred.tolist()]
        
        try:
            with open(TEST_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(results)
        except Exception as e:
            print("Error writing CSV:", e)

def placebo_test_cim_bign(result_log_name, d, n, T0):
    TEST_FILE = 'resultLogsnba/' + result_log_name + '.csv'

    header = ["seed", "T", "T0", "N1", "n", "d",
              "trueTarget", "pred_cim", "cim_posterior_lower", "cim_posterior_upper"]

    with open(TEST_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    T = 192 
    N1 = 1000
    game_data, year_data = get_nba_data()
    num_reps = NUM_REPS 
    seed_error = 1000
    for i in range(num_reps):
        print(f"Iteration {i+1} / {num_reps}")
        seed = i
        set_seed(seed)
        set_tf_seed(seed)
        max_retries = 10
        seed_error = 1000
        attempt = 0
        success = False
        while attempt < max_retries and not success:
            try: 
                Y = torch.from_numpy(make_placebo_Y(game_data, year_data, n).T).to(torch.float32)
                success = True
            except Exception as e:
                print(f"Attempt {attempt + 1} failed with seed {seed}. Error: {e}")
                seed = seed_error + seed
                attempt += 1
                set_seed(seed)
                set_tf_seed(seed)
        if not success:
            raise RuntimeError("getting donor failed after multiple retries")

        mean_time_series = Y[:, 1:].mean(dim=1)
        Y_centered = Y - mean_time_series.view(-1, 1)

        impact, cim_pred, cim_posterior_lower, cim_posterior_upper = cim_prediction(Y_centered, T0)
        cim_pred = cim_pred + mean_time_series.detach().numpy()

        target_true = Y[:, 0].detach().numpy()

        identifiers = [seed, T, T0, N1, n, d]
        results = identifiers + [target_true.tolist(), 
                                 cim_pred.tolist(),
                                 cim_posterior_lower.tolist(),
                                 cim_posterior_upper.tolist(), 
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

    placebo_test_rsc_params("nba_placebo_coeff_rsc_n96_d5_T077", 5, 96, 77)

    placebo_test_rsc_sc("nba_placebo_test_rsc_sc_n24_d5_T096_aistats", 5, 24, 96)
    placebo_test_rsc_sc("nba_placebo_test_rsc_sc_n48_d5_T096_aistats", 5, 48, 96)
    placebo_test_rsc_sc("nba_placebo_test_rsc_sc_n96_d5_T096_aistats", 5, 96, 96)
    placebo_test_rsc_sc("nba_placebo_test_rsc_sc_n192_d5_T096_aistats", 5, 192, 96)
    placebo_test_rsc_sc_bign("nba_placebo_test_rsc_sc_n384_d5_T096_aistats", 5, 384, 96)

    placebo_test_tasc("nba_placebo_test_tasc_n24_d5_T096_aistats", 5, 24, 96)
    placebo_test_tasc("nba_placebo_test_tasc_n48_d5_T096_aistats", 5, 48, 96)
    placebo_test_tasc("nba_placebo_test_tasc_n96_d5_T096_aistats", 5, 96, 96)
    placebo_test_tasc("nba_placebo_test_tasc_n192_d5_T096_aistats", 5, 192, 96)
    placebo_test_tasc_bign("nba_placebo_test_tasc_n384_d5_T096_aistats", 5, 384, 96)

    ## Now working with tensorflow so use the virtual environment
    ## It is best to run the below tests seperately in an environment with tensorflow install. 
    ## We found on some machines importing tensorflow and causalimpact impacted classic SC performance
    ## TASC and RSC are not impacted by causalimpact and tensorflow library included 

    # import tensorflow as tf 
    # import tensorflow_probability as tfp
    # import causalimpact

    # placebo_test_cim("nba_placebo_test_cim_n24_d5_T096_aistats", 5, 24, 96)
    # placebo_test_cim("nba_placebo_test_cim_n48_d5_T096_aistats", 5, 48, 96)
    # placebo_test_cim("nba_placebo_test_cim_n96_d5_T096_aistats", 5, 96, 96)
    # placebo_test_cim("nba_placebo_test_cim_n192_d5_T096_aistats", 5, 192, 96)
    # placebo_test_cim_bign("nba_placebo_test_cim_n384_d5_T096_aistats", 5, 384, 96)
