from matplotlib import pyplot as plt 

import pyreadr
import numpy as np
import pandas as pd 
import copy 
from scipy.optimize import fmin_slsqp
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from matrix import Matrix 
from synthetic_control import SyntheticControl

from tasc import TimeAwareSC 
import torch 
import random 
from torch import nn
import csv

## Comment these in when doing tests involving CIM
## Importing these can have dependency issues with classic SC so best to run SC tests without imported
# import tensorflow as tf 
# import tensorflow_probability as tfp
# import causalimpact 

START_TIME = 1970
INTERVENTION_TIME = 1989
STOP_TIME = 2001

def extract_predictor_vec(df_outcome, df_predictors, state, targetState):
    df_outcome_state = df_outcome[df_outcome['LocationDesc'] == state]
    cigsale88_predictor = df_outcome_state['1988'].item()
    cigsale80_predictor = df_outcome_state['1980'].item()
    cigsale75_predictor = df_outcome_state['1975'].item()
    
    state_id_predictors_df = df_outcome[df_outcome['LocationDesc'] == targetState].index.item() + 1 
    df_predictors_state = df_predictors[df_predictors['state'] == state_id_predictors_df]
    beer_predictor = df_predictors_state.loc[(df_predictors_state['year'] >= 1984) & (df_predictors_state['year'] < INTERVENTION_TIME), 'beer'].mean()
    age15to24_predictor = df_predictors_state.loc[(df_predictors_state['year'] >= 1980) & (df_predictors_state['year'] < INTERVENTION_TIME), 'age15to24'].mean()*100  
    retprice_predictor = df_predictors_state.loc[(df_predictors_state['year'] >= 1980) & (df_predictors_state['year'] < INTERVENTION_TIME), 'retprice'].mean()
    lnincome_predictor = df_predictors_state.loc[(df_predictors_state['year'] >= 1980) & (df_predictors_state['year'] < INTERVENTION_TIME), 'lnincome'].mean()
    
    return np.array([lnincome_predictor, age15to24_predictor, retprice_predictor, beer_predictor,  
                     cigsale88_predictor, cigsale80_predictor, cigsale75_predictor]).reshape(-1,1)


def get_predictors(df_outcome, df_predictors, targetState):
    control_predictors = []
    for state in df_outcome['LocationDesc'].unique():
        state_predictor_vec = extract_predictor_vec(df_outcome, df_predictors, state, targetState)
        if state == targetState:
            ca_predictors = state_predictor_vec
        else:
            control_predictors += [state_predictor_vec]

    control_predictors = np.hstack(control_predictors) 

    X0 = control_predictors
    X1 = ca_predictors
    return X0, X1

def w_mse(w, v, x0, x1): return mean_squared_error(x1, x0.dot(w), sample_weight=v)

def w_mse(w, v, x0, x1): return mean_squared_error(x1, x0.dot(w), sample_weight=v)

def w_constraint(w, v, x0, x1): return np.sum(w) - 1

def v_constraint(V, W, X0, X1, Z0, Z1): return np.sum(V) - 1

def fun_w(w, v, x0, x1): return fmin_slsqp(w_mse, w, bounds=[(0.0, 1.0)]*len(w), f_eqcons=w_constraint, 
                                           args=(v, x0, x1), disp=False, full_output=True)[0]

def fun_v(v, w, x0, x1, z0, z1): return mean_squared_error(z1, z0.dot(fun_w(w, v, x0, x1)))

def solve_synthetic_control(X0, X1, Z0, Z1, Y0):
    k,j = X0.shape
    V0 = 1/k*np.ones(k)
    W0 = 1/j*np.zeros(j).transpose()
    V = fmin_slsqp(fun_v, V0, args=(W0, X0, X1, Z0, Z1), bounds=[(0.0, 1.0)]*len(V0), disp=True, f_eqcons=v_constraint, acc=1e-6)
    W = fun_w(W0, V, X0, X1)
    return V, W

def get_components(df_outcome, targetState):
    df_outcome_ca = df_outcome.loc[df_outcome['LocationDesc'] == targetState, :]
    df_outcome_control = df_outcome.loc[df_outcome['LocationDesc'] != targetState, :]

    ca_outcomes_pre = df_outcome_ca.loc[:,[str(i) for i in list(range(START_TIME, INTERVENTION_TIME))]].values.reshape(-1,1)
    control_outcomes_pre = df_outcome_control.loc[:,[str(i) for i in list(range(START_TIME, INTERVENTION_TIME))]].values.transpose()

    ca_outcomes_post = df_outcome_ca.loc[:,[str(i) for i in list(range(INTERVENTION_TIME, STOP_TIME))]].values.reshape(-1,1)
    control_outcomes_post = df_outcome_control.loc[:,[str(i) for i in list(range(INTERVENTION_TIME, STOP_TIME))]].values.transpose()

    Z0 = control_outcomes_pre
    Z1 = ca_outcomes_pre
    Y0 = control_outcomes_post
    Y1 = ca_outcomes_post

    return Z0, Z1, Y0, Y1

def solve_synthetic_control(X0, X1, Z0, Z1, Y0):
    k,j = X0.shape
    V0 = 1/k*np.ones(k)
    W0 = 1/j*np.zeros(j).transpose()
    V = fmin_slsqp(fun_v, V0, args=(W0, X0, X1, Z0, Z1), bounds=[(0.0, 1.0)]*len(V0), disp=True, f_eqcons=v_constraint, acc=1e-6)
    W = fun_w(W0, V, X0, X1)
    return V, W

def prepare_data_placebo(bad_states=None):
    df_outcome_raw = pd.read_csv('../../Data/prop99.csv')
    df_outcome_raw = df_outcome_raw[df_outcome_raw['SubMeasureDesc'] == 'Cigarette Consumption (Pack Sales Per Capita)']
    df_outcome = pd.DataFrame(df_outcome_raw.pivot_table(values='Data_Value', index='LocationDesc', columns=['Year']).to_records())

    rda_predictors = pyreadr.read_r('../../Data/smoking.rda')
    df_predictors = pd.DataFrame(list(rda_predictors.values())[0])
    if(bad_states == None):
        bad_states = ['Massachusetts', 'Arizona', 'Oregon', 'Florida', 'Alaska', 'Hawaii', 'Maryland', 
                    'Michigan', 'New Jersey', 'New York', 'Washington', 'District of Columbia', 'California']

    df_outcome.drop(df_outcome[df_outcome['LocationDesc'].isin(bad_states)].index, inplace=True)
    df_outcome = df_outcome.reset_index()
    df_outcome = df_outcome.rename(columns={'index': 'org_index'})
    return df_outcome, df_predictors

def prepare_Y_data(targetState, bad_states=None):
    df_outcome, _ = prepare_data_placebo(bad_states)
    Z0, Z1, Y0, Y1 = get_components(df_outcome, targetState)
    ca_row = np.concatenate([Z1.flatten(), Y1.flatten()])
    donor_rows = np.vstack([Z0, Y0]).T
    full_data = np.vstack([ca_row, donor_rows])
    Y = full_data.T
    return Y

def true_prediction(targetState, bad_states=None):
    Y = prepare_Y_data(targetState, bad_states)
    return Y[:, 0]

def kalman_prediction(targetState, T0, N1, d, seed=None, bad_states=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Y = prepare_Y_data(targetState, bad_states)
    Y_mask = Y.copy()
    Y_mask[T0:, 0] = [None] * (STOP_TIME - INTERVENTION_TIME)
    Y_tor = torch.from_numpy(Y).clone().to(torch.float32)
    model = TimeAwareSC(Y=Y_tor.T.to(device), d=d, device=device, dtype=torch.float32)
    model.initialize_theta(method='naive', random_seed=seed)
    model.T0 = T0
    model.em_pre(T0=T0, N1=N1)
    log_like = model.log_likelihood(T=T0).item()
    with torch.no_grad():
        target_pred, donor_pred, tasc_target_var_estimates = model.make_prediction()
        R_tasc = np.diag(model.R.cpu().numpy())

    return log_like, target_pred.detach().numpy(), tasc_target_var_estimates, R_tasc

def cim_prediction(targetState, T0, bad_states=None):
    Y = prepare_Y_data(targetState, bad_states)
    Y_mask = Y.copy()
    T, N = Y.shape
    Y_mask[T0:, 0] = [None] * (STOP_TIME - INTERVENTION_TIME)
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

def synthetic_control_prediction(targetState, T0, bad_states=None):
    Y = prepare_Y_data(targetState, bad_states)
    Y_mask = Y.copy()
    Y_mask[T0:, 0] = [None] * (STOP_TIME - INTERVENTION_TIME)
    df = pd.DataFrame(Y_mask)
    M = Matrix(df, T0, target_name=0)
    sys = SyntheticControl()
    sys.fit(M.pre_donor, M.pre_target, method='simplex')
    ca_pred_sc = np.array(sys.predict(M.donor)).squeeze()
    return ca_pred_sc

def rsc_prediction(targetState, T0, d, rscmethod='ridge', rsclmbda=0.1, bad_states=None):
    Y = prepare_Y_data(targetState, bad_states)
    Y_mask = Y.copy()
    Y_mask[T0:, 0] = [None] * (STOP_TIME - INTERVENTION_TIME)
    df = pd.DataFrame(Y_mask)
    M = Matrix(df, T0, target_name=0)
    M.denoise(num_sv=d)
    sys = SyntheticControl()
    sys.fit(M.pre_donor, M.pre_target, method=rscmethod, lmbda=rsclmbda)
    ca_pred_rsc = np.array(sys.predict(M.donor)).squeeze()
    return ca_pred_rsc


def main():
    run_vary_d_test("prop99_vary_d_results_aistats")
    run_california_test_no_cim("prop99_CaliforniaResults_aistats_nocim")
    run_residual_test()

    ## These tests run with causal impact imports (at top) included
    # run_placebo_test_just_cim("vary_d_results_just_cim_aistats")
    # run_california_test("prop99_CaliforniaResults_aistats_includingcim")
    # run_residual_test_just_cim()
    # run_california_test_just_cim("prop99_CaliforniaResults_aistats_just_cim")

    print("Done")

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
    # tf.random.set_seed(seed)

def set_tf_seed(seed: int = 42):
    tf.random.set_seed(seed)

def run_placebo_test_just_cim(result_log_name):
    seed = 1
    T0 = INTERVENTION_TIME - START_TIME
    T = STOP_TIME - START_TIME
    N1 = 1000
    TEST_FILE = 'resultLogsprop99/' + result_log_name + '.csv'

    header = ["seed", "targetState", "T", "T0", "N1", "d_rsc", "d_kalman", 
              "trueTarget", "pred_cim", "cim_posterior_lower", "cim_posterior_upper" ]

    with open(TEST_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    df_outcome, _ = prepare_data_placebo()
    all_states = df_outcome['LocationDesc'].unique()

    #Running this with different d is unnecessary doing it for plotting simplicity
    for d_est in [2, 4, 8, 16]:
        for targetState in all_states: 
            print(f"d_est = {d_est}, targetState = {targetState}")
            seed += 1
            set_seed(seed)
            set_tf_seed(seed)
            target_true = true_prediction(targetState)

            impact, cim_pred, cim_posterior_lower, cim_posterior_upper = cim_prediction(targetState, T0)

            rsc_d = d_est
            kalman_d = d_est
            identifiers = [seed, targetState, T, T0, N1, rsc_d, kalman_d]
            results = identifiers + [target_true.tolist(), 
                                    cim_pred.tolist(),
                                    cim_posterior_lower.tolist(),
                                    cim_posterior_upper.tolist() ]


            try:
                with open(TEST_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(results)
            except Exception as e:
                print("Error writing CSV:", e)

def run_vary_d_test(result_log_name):
    seed = 1
    T0 = INTERVENTION_TIME - START_TIME
    T = STOP_TIME - START_TIME
    N1 = 1000
    TEST_FILE = 'resultLogsprop99/' + result_log_name + '.csv'

    header = ["seed", "targetState", "T", "T0", "N1", "d_rsc", "d_kalman", "loglikelihood",
              "trueTarget", "pred_tasc", "pred_rsc", "pred_sc",  
               "tasc_target_var_estimates", "R_tasc"]

    with open(TEST_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    df_outcome, _ = prepare_data_placebo()
    all_states = df_outcome['LocationDesc'].unique()
    for d_est in [2, 4, 8, 16]:
        for targetState in all_states: 
            print(f"d_est = {d_est}, targetState = {targetState}")
            seed += 1
            set_seed(seed)
            target_true = true_prediction(targetState)
            pred_rsc = rsc_prediction(targetState, T0, d_est, rscmethod='ridge', rsclmbda=0.1)

            max_retries = 50
            attempt = 0
            success = False
            seed_error = seed
            seed_error_mv = 1000
            while attempt < max_retries and not success:
                try:
                    logp, pred_tasc, tasc_target_var_estimates, R_tasc = kalman_prediction(targetState, T0, N1, d_est, seed_error)
                    success = True
                
                except Exception as e:
                    print(f"Attempt {attempt+1} failed with tascinit seed {seed_error}. Reinitializing for tasc")
                    seed_error += seed_error_mv 
            
            if not success:
                raise RuntimeError("TASC prediction failed after multiple tries")

            

            sc_pred = synthetic_control_prediction(targetState, T0)

            rsc_d = d_est
            kalman_d = d_est
            identifiers = [seed, targetState, T, T0, N1, rsc_d, kalman_d]
            results = identifiers + [logp, target_true.tolist(), 
                                    pred_tasc.tolist(), 
                                    pred_rsc.tolist(), 
                                    sc_pred.tolist(),
                                    tasc_target_var_estimates.tolist(), 
                                    R_tasc.tolist()]


            try:
                with open(TEST_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(results)
            except Exception as e:
                print("Error writing CSV:", e)

def run_california_test_no_cim(result_log_name):
    seed = 1
    set_seed(seed)
    T0 = INTERVENTION_TIME - START_TIME
    T = STOP_TIME - START_TIME
    N1 = 1000
    rsc_d = 2
    kalman_d = 2
    TEST_FILE = 'resultLogsprop99/' + result_log_name + '.csv'

    bad_states = ['Massachusetts', 'Arizona', 'Oregon', 'Florida', 'Alaska', 'Hawaii', 'Maryland', 
                'Michigan', 'New Jersey', 'New York', 'Washington', 'District of Columbia']
    df_outcome, _ = prepare_data_placebo(bad_states)
    all_states = df_outcome['LocationDesc'].unique()
    targetState = 'California'

    target_true = true_prediction(targetState, bad_states)
    pred_rsc = rsc_prediction(targetState, T0, rsc_d, rscmethod="ridge", rsclmbda=0.1, bad_states=bad_states)
    sc_pred = synthetic_control_prediction(targetState, T0, bad_states)
    logp, pred_tasc, tasc_target_var_estimates, R_tasc = kalman_prediction(targetState, T0, N1, kalman_d, seed, bad_states)

    header = ["seed", "targetState", "T", "T0", "N1", "d_rsc", "d_kalman", "loglikelihood",
              "trueTarget", "pred_tasc", "pred_rsc", "pred_sc"]

    with open(TEST_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    identifiers = [seed, targetState, T, T0, N1, rsc_d, kalman_d]
    results = identifiers + [logp, target_true.tolist(), 
                             pred_tasc.tolist(), 
                             pred_rsc.tolist(), 
                             sc_pred.tolist()
                             ]

    try:
        with open(TEST_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(header)
            writer.writerow(results)
    except Exception as e:
        print("Error writing CSV:", e)

def run_california_test(result_log_name):
    seed = 1
    set_seed(seed)
    set_tf_seed(seed)
    T0 = INTERVENTION_TIME - START_TIME
    T = STOP_TIME - START_TIME
    N1 = 1000
    rsc_d = 2
    kalman_d = 2
    TEST_FILE = 'resultLogsprop99/' + result_log_name + '.csv'

    bad_states = ['Massachusetts', 'Arizona', 'Oregon', 'Florida', 'Alaska', 'Hawaii', 'Maryland', 
                'Michigan', 'New Jersey', 'New York', 'Washington', 'District of Columbia']
    df_outcome, _ = prepare_data_placebo(bad_states)
    all_states = df_outcome['LocationDesc'].unique()
    targetState = 'California'

    target_true = true_prediction(targetState, bad_states)
    pred_rsc = rsc_prediction(targetState, T0, rsc_d, rscmethod="ridge", rsclmbda=0.1, bad_states=bad_states)
    sc_pred = synthetic_control_prediction(targetState, T0, bad_states)

    logp, pred_tasc, tasc_target_var_estimates, R_tasc = kalman_prediction(targetState, T0, N1, kalman_d, seed, bad_states)
    impact, cim_pred, cim_posterior_lower, cim_posterior_upper = cim_prediction(targetState, T0, bad_states=bad_states)

    header = ["seed", "targetState", "T", "T0", "N1", "d_rsc", "d_kalman", "loglikelihood",
              "trueTarget", "pred_tasc", "pred_rsc", "pred_sc", "pred_cim", 
              "cim_posterior_lower", "cim_posterior_upper", "tasc_target_var_estimates", "R_tasc"]



    with open(TEST_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    identifiers = [seed, targetState, T, T0, N1, rsc_d, kalman_d]
    results = identifiers + [logp, target_true.tolist(), 
                             pred_tasc.tolist(), 
                             pred_rsc.tolist(), 
                             sc_pred.tolist(),
                             cim_pred.tolist(),
                             cim_posterior_lower.tolist(),
                             cim_posterior_upper.tolist(),
                             tasc_target_var_estimates.tolist(), 
                             R_tasc.tolist()]

    try:
        with open(TEST_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(header)
            writer.writerow(results)
    except Exception as e:
        print("Error writing CSV:", e)

def run_california_test_just_cim(result_log_name):
    seed = 1 
    set_seed(seed)
    set_tf_seed(seed)
    T0 = INTERVENTION_TIME - START_TIME
    T = STOP_TIME - START_TIME
    N1 = 1000
    rsc_d = 2
    kalman_d = 2
    TEST_FILE = 'resultLogsprop99/' + result_log_name + '.csv'

    bad_states = ['Massachusetts', 'Arizona', 'Oregon', 'Florida', 'Alaska', 'Hawaii', 'Maryland', 
                'Michigan', 'New Jersey', 'New York', 'Washington', 'District of Columbia']
    df_outcome, _ = prepare_data_placebo(bad_states)
    all_states = df_outcome['LocationDesc'].unique()
    targetState = 'California'

    target_true = true_prediction(targetState, bad_states)
    impact, cim_pred, cim_posterior_lower, cim_posterior_upper = cim_prediction(targetState, T0, bad_states=bad_states)

    header = ["seed", "targetState", "T", "T0", "N1", "d_rsc", "d_kalman", 
              "trueTarget", "pred_cim", 
              "cim_posterior_lower", "cim_posterior_upper"]

    with open(TEST_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    identifiers = [seed, targetState, T, T0, N1, rsc_d, kalman_d]
    results = identifiers + [target_true.tolist(), 
                             cim_pred.tolist(),
                             cim_posterior_lower.tolist(),
                             cim_posterior_upper.tolist() ]

    try:
        with open(TEST_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(header)
            writer.writerow(results)
    except Exception as e:
        print("Error writing CSV:", e)

def run_residual_test_just_cim():
    T0 = INTERVENTION_TIME - START_TIME
    N1 = 1000
    N2 = 10
    rsc_d = 2
    kalman_d = 2
    TEST_FILE = "resultLogsprop99/prop99residualResults_just_cim_aistats.csv"
    seed = 1
    set_seed(seed)
    set_tf_seed(seed)
    df_outcome, _ = prepare_data_placebo()
    all_states = df_outcome['LocationDesc'].unique()

    header = ["targetState", "N1", "N2", "d_rsc", "d_kalman"] + \
            [f"a{i}" for i in range(31)] + \
            [f"b{i}" for i in range(31)] 

    with open(TEST_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    for targetState in all_states: 
        seed += 1 
        set_tf_seed(seed)
        set_seed(seed)
        target_true = true_prediction(targetState)
        impact, cim_pred, cim_posterior_lower, cim_posterior_upper = cim_prediction(targetState, T0)


        header = ["targetState", "N1", "N2", "d_rsc", "d_kalman"] + \
                [f"a{i}" for i in range(31)] + \
                [f"b{i}" for i in range(31)] 
        
        row = [targetState, N1, N2, rsc_d, kalman_d] + list(target_true) + list(cim_pred) 

        try:
                with open(TEST_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    if f.tell() == 0:
                        writer.writerow(header)
                    writer.writerow(row)
        except Exception as e:
                print("Error writing CSV:", e)

    bad_states = ['Massachusetts', 'Arizona', 'Oregon', 'Florida', 'Alaska', 'Hawaii', 'Maryland', 
                'Michigan', 'New Jersey', 'New York', 'Washington', 'District of Columbia']
    df_outcome, _ = prepare_data_placebo(bad_states)
    all_states = df_outcome['LocationDesc'].unique()
    targetState = 'California'

    target_true = true_prediction(targetState, bad_states)
    impact, cim_pred, cim_posterior_lower, cim_posterior_upper = cim_prediction(targetState, T0, bad_states=bad_states)

    header = ["targetState", "N1", "N2", "d_rsc", "d_kalman"] + \
        [f"a{i}" for i in range(31)] + \
        [f"b{i}" for i in range(31)] 

    row = [targetState, N1, N2, rsc_d, kalman_d] + list(target_true) + list(cim_pred) 

    try:
        with open(TEST_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(header)
            writer.writerow(row)
    except Exception as e:
        print("Error writing CSV:", e)

def run_residual_test():
    T0 = INTERVENTION_TIME - START_TIME
    N1 = 1000
    N2 = 10
    rsc_d = 2
    kalman_d = 2
    TEST_FILE = "resultLogsprop99/prop99residualResults.csv"
    seed = 1
    set_seed(seed)
    header = ["targetState", "N1", "N2", "d_rsc", "d_kalman"] + \
            [f"a{i}" for i in range(31)] + \
            [f"b{i}" for i in range(31)] + \
            [f"c{i}" for i in range(31)] + \
            [f"d{i}" for i in range(31)] 
    with open(TEST_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    df_outcome, _ = prepare_data_placebo()
    all_states = df_outcome['LocationDesc'].unique()
    for targetState in all_states: 
        print(f"target state: {targetState}")
        seed += 1
        set_seed(seed)
        target_true = true_prediction(targetState)
        pred_rsc = rsc_prediction(targetState, T0, rsc_d, rscmethod='ridge', rsclmbda=0.1)
        logp, pred_tasc, tasc_target_var_estimates, R_tasc = kalman_prediction(targetState, T0, N1, kalman_d, seed)
        sc_pred = synthetic_control_prediction(targetState, T0)

        header = ["targetState", "N1", "N2", "d_rsc", "d_kalman"] + \
                [f"a{i}" for i in range(31)] + \
                [f"b{i}" for i in range(31)] + \
                [f"c{i}" for i in range(31)] + \
                [f"d{i}" for i in range(31)] 
        
        row = [targetState, N1, N2, rsc_d, kalman_d] + list(target_true) + list(pred_tasc) + list(sc_pred) + list(pred_rsc)

        try:
                with open(TEST_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    if f.tell() == 0:
                        writer.writerow(header)
                    writer.writerow(row)
        except Exception as e:
                print("Error writing CSV:", e)

    bad_states = ['Massachusetts', 'Arizona', 'Oregon', 'Florida', 'Alaska', 'Hawaii', 'Maryland', 
                'Michigan', 'New Jersey', 'New York', 'Washington', 'District of Columbia']
    df_outcome, _ = prepare_data_placebo(bad_states)
    all_states = df_outcome['LocationDesc'].unique()
    targetState = 'California'

    target_true = true_prediction(targetState, bad_states)
    pred_rsc = rsc_prediction(targetState, T0, rsc_d, rscmethod='ridge', rsclmbda=0.1, bad_states=bad_states)
    logp, pred_tasc, tasc_target_var_estimates, R_tasc = kalman_prediction(targetState, T0, N1, kalman_d, seed, bad_states=bad_states)
    sc_pred = synthetic_control_prediction(targetState, T0, bad_states=bad_states)

    header = ["targetState", "N1", "N2", "d_rsc", "d_kalman"] + \
        [f"a{i}" for i in range(31)] + \
        [f"b{i}" for i in range(31)] + \
        [f"c{i}" for i in range(31)] + \
        [f"d{i}" for i in range(31)] 

    row = [targetState, N1, N2, rsc_d, kalman_d] + list(target_true) + list(pred_tasc) + list(sc_pred) + list(pred_rsc)

    try:
        with open(TEST_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(header)
            writer.writerow(row)
    except Exception as e:
        print("Error writing CSV:", e)


if __name__ == "__main__":
    main()