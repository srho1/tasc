import math
import numpy as np
import pandas as pd
import warnings

from numpy import random
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import cauchy

from matrix import Matrix
from synthetic_control import SyntheticControl
from syslibutils import *


def generate_rank_1_matrix(n, m, noise_level, smooth=True):
    u = random.rand(n)
    # v = random.rand(m)
    if smooth:
        # i will add something here to ensure lipschitz continuity
        window_size = 5
        v = random.beta(2, 2, m + window_size - 1)
        window = np.ones(window_size) / window_size
        # v = np.concatenate((v, np.zeros(window_size-1)))
        v = np.convolve(v, window, mode="valid")
    else:
        v = random.rand(m)
    dataset = pd.DataFrame(np.outer(u, v))
    dataset += random.normal(0, noise_level, (n, m))
    return dataset.T


def generate_rank_k_matrix(n, m, k, noise_level, smooth=True):
    u = random.rand(n, k)
    if smooth:
        # i will add something here to ensure lipschitz continuity
        window_size = 5
        v_k = np.array([])
        for _ in range(k):
            v = random.beta(2, 2, m + window_size - 1)
            window = np.ones(window_size) / window_size
            # v = np.concatenate((v, np.zeros(window_size-1)))
            v = np.convolve(v, window, mode="valid")
            v_k = np.concatenate((v_k, v))
        v_k = v_k.reshape(k, m)
    else:
        v_k = random.rand(k, m)
    dataset = pd.DataFrame(u @ v_k)
    dataset += random.normal(0, noise_level, (n, m))
    return dataset.T


def generate_sine_wave(alpha, omega, phi, noise_level, num_time):
    time = np.arange(num_time) * 10 * np.pi  # fast forward time by 10*pi
    signal = alpha * np.sin(2 * np.pi * omega * time / 360 + phi)
    noise = 0
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, num_time)
    return signal + noise


def zero_out_fraction(
    df: pd.DataFrame, p: float = 0.9
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Zero out a fraction of entries in a pandas DataFrame.

    Args:
    df (pd.DataFrame): Input DataFrame to modify
    p (float): Fraction of observed entries (default: 0.9)

    Returns:
    tuple: (modified_df, zero_indices)
    """
    # Ensure p is between 0 and 1
    if not 0 <= p <= 1:
        raise ValueError("p must be between 0 and 1")

    # Get the shape of the DataFrame
    rows, cols = df.shape

    # Total number of elements in the DataFrame
    total_elements = rows * cols

    # Calculate how many elements to set to zero
    num_zeros = int((1 - p) * total_elements)

    # Create a flat array of all indices in the DataFrame
    all_indices = np.arange(total_elements)

    # Randomly select indices to zero out
    zero_indices = np.random.choice(all_indices, num_zeros, replace=False)

    # Create a copy of the DataFrame to avoid modifying the original
    modified_df = df.copy()

    # Convert DataFrame to numpy array, set chosen indices to zero, then convert back to DataFrame
    modified_values = modified_df.values
    modified_values.flat[zero_indices] = 0
    modified_df = pd.DataFrame(modified_values, index=df.index, columns=df.columns)

    return modified_df, zero_indices


def generate_linear_dataset(
    num_samples, num_time, noise_level, mean=3, std=1, noise_type="normal"
):
    # outputs T by n matrix

    times = np.reshape(np.arange(num_time), (1, num_time))
    times = times / num_time
    slope = random.normal(mean, std, num_samples)
    intercept = random.uniform(-1, 1, num_samples)
    data = np.reshape(slope, (num_samples, 1)) @ times
    if noise_type == "normal":
        noise = random.normal(0, noise_level, (num_samples, num_time))
        data += np.tile(intercept, (num_time, 1)).T  # n by T
        data += noise
    if noise_type == "cauchy":
        noise = cauchy.rvs(scale=noise_level, size=(num_samples, num_time))
        data += np.tile(intercept, (num_time, 1)).T  # n by T
        data += noise
    # if noise_type == "sparse":
    #     noise = random.normal(0, noise_level, (num_samples, num_time)) # gaussian noise
    #     data += np.tile(intercept, (num_time, 1)).T # n by T
    #     data = zero_out_fraction(data, noise_level) # and then zero out some fraction of the data

    return pd.DataFrame(data.T)  # T by n dataframe


# def generate_sine_dataset(num_samples, num_time, alpha, omega, phi, noise_level, param_noise = 0.05):
#     # alpha: magnitude, omega: frequency, phi: delay
#     dataset = pd.DataFrame()
#     for i in range(num_samples):
#         alpha_noisy = alpha + random.normal(0, param_noise) # noisy magnitude
#         omega_noisy = omega + random.normal(0, param_noise) # noisy frequency
#         phi_noisy = phi + random.normal(0, param_noise) # noisy delay
#         y = generate_sine_wave(alpha_noisy, omega_noisy, phi_noisy, noise_level, int(num_time*1.2))
#         dataset[i] = y[int(0.2*num_time):] # remove the first 20% of the data
#     return dataset


# def generate_additive_sine_dataset(num_samples, num_time, noise_level, num_signals,
#                                     param_noise=0.05, approx_low_rank=True):
#     # alpha: magnitude, omega: frequency, phi: delay
#     final_dataset = np.zeros((num_time, num_samples))

#     for _ in range(num_signals):
#         dataset = pd.DataFrame()

#         alpha = random.beta(2,2,1)
#         omega = math.pi/random.uniform(low=1, high=8, size=1)
#         phi = random.normal(0,1)

#         for i in range(num_samples):
#             weight = random.uniform(0, 1, 1)
#             alpha_noisy = alpha + random.normal(0, param_noise) # noisy magnitude
#             omega_noisy = omega + random.normal(0, param_noise) # noisy frequency
#             phi_noisy = phi + random.normal(0, param_noise) # noisy delay
#             y = generate_sine_wave(alpha_noisy, omega_noisy, phi_noisy, 0, int(num_time*1.2))
#             dataset[i] = weight * y[int(0.2*num_time):] # remove the first 20% of the data
#         final_dataset = final_dataset + dataset
#     if approx_low_rank:
#         final_dataset = make_approx_low_rank(final_dataset, num_signals)
#     # add observational noise
#     final_dataset += random.normal(0, noise_level, (num_time, num_samples))
#     return pd.DataFrame(final_dataset)

# def make_approx_low_rank(dataset, k=5):
#     u, s, vh = np.linalg.svd(dataset, full_matrices=False)
#     s[k:] = 0.1 * s[k:]
#     return u @ np.diag(s) @ vh


# def generate_new_sine_dataset(
#     num_samples,
#     num_time,
#     noise_level,
#     num_signals,
#     param_noise=0.05,
#     approx_low_rank=True,
# ):
#     # alpha: magnitude, omega: frequency, phi: delay
#     # outputs T by n matrix
#     basis_vectors = np.zeros((num_signals, num_time))
#     for i in range(num_signals):
#         alpha = random.beta(2, 2, 1)
#         omega = random.uniform(low=1, high=10, size=1)
#         phi = random.normal(0, 1)
#         y = generate_sine_wave(alpha, omega, phi, 0, int(num_time * 1.2))
#         basis_vectors[i] = y[int(0.2 * num_time) :]  # remove the first 20% of the data

#     weight_vectors = random.uniform(0, 1, (num_samples, num_signals))

#     final_dataset = weight_vectors @ basis_vectors
#     final_dataset += random.normal(0, noise_level, (num_samples, num_time))
#     return pd.DataFrame(final_dataset).T


def generate_new_sine_dataset(
    num_samples,
    num_time,
    noise_level,
    num_signals,
    low=1,  # for omega uniform low
    high=10,  # for omega uniform high
    alpha=None,
    omega=None,
    phi=None,
):
    # alpha: magnitude, omega: frequency, phi: delay
    # alpha, omega, phi should be lists of size num_signals
    # outputs T by n matrix
    basis_vectors = np.zeros((num_signals, num_time))
    for i in range(num_signals):
        alpha = random.beta(2, 2, 1) or alpha[i]
        omega = random.uniform(low=low, high=high, size=1) or omega[i]
        phi = random.normal(0, 1) or phi[i]
        y = generate_sine_wave(alpha, omega, phi, 0, int(num_time * 1.2))
        basis_vectors[i] = y[int(0.2 * num_time) :]  # remove the first 20% of the data

    weight_vectors = random.uniform(0, 1, (num_samples, num_signals))

    final_dataset = weight_vectors @ basis_vectors
    final_dataset += random.normal(0, noise_level, (num_samples, num_time))
    return pd.DataFrame(final_dataset).T


def generate_sine_dataset_A(
    num_samples,
    num_time,
    noise_level,
    num_signals,
    noise_type="normal",
):
    # alpha: magnitude, omega: frequency, phi: delay
    # outputs T by n matrix
    basis_vectors = np.zeros((num_signals, num_time))
    for i in range(num_signals):
        # alpha = random.normal(1,1)
        alpha = random.beta(2, 2, 1)
        omega = random.uniform(low=1, high=3, size=1)
        phi = random.normal(0, 1)
        y = generate_sine_wave(alpha, omega, phi, 0, int(num_time * 1.2))
        basis_vectors[i] = y[int(0.2 * num_time) :]  # remove the first 20% of the data

    weight_vectors = random.uniform(0, 1, (num_samples, num_signals))

    final_dataset = weight_vectors @ basis_vectors

    if noise_type == "normal":
        final_dataset += random.normal(0, noise_level, (num_samples, num_time))
    elif noise_type == "cauchy":
        final_dataset += cauchy.rvs(scale=noise_level, size=(num_samples, num_time))
    else:
        raise ValueError("Invalid noise type")
    return pd.DataFrame(final_dataset).T


def generate_sine_dataset_B(
    num_samples,
    num_time,
    noise_level,
    num_signals,
    noise_type="normal",
):
    # alpha: magnitude, omega: frequency, phi: delay
    # outputs T by n matrix
    basis_vectors = np.zeros((num_signals, num_time))
    for i in range(num_signals):
        # alpha = random.normal(3,1)
        alpha = random.beta(2, 5, 1)
        omega = random.uniform(low=3, high=6, size=1)
        phi = random.normal(0, 1)
        y = generate_sine_wave(alpha, omega, phi, 0, int(num_time * 1.2))
        basis_vectors[i] = y[int(0.2 * num_time) :]  # remove the first 20% of the data

    weight_vectors = random.uniform(0, 1, (num_samples, num_signals))

    final_dataset = weight_vectors @ basis_vectors

    if noise_type == "normal":
        final_dataset += random.normal(0, noise_level, (num_samples, num_time))
    elif noise_type == "cauchy":
        final_dataset += cauchy.rvs(scale=noise_level, size=(num_samples, num_time))
    else:
        raise ValueError("Invalid noise type")
    return pd.DataFrame(final_dataset).T


def make_approx_low_rank(dataset, k=5):
    u, s, vh = np.linalg.svd(dataset, full_matrices=False)
    s[k:] = 0.1 * s[k:]
    return u @ np.diag(s) @ vh


def get_gamma_pi(dataset_1, dataset_2, verbose=False):
    # dataset comes in as T by n, so transpose before svd
    cluster_vh = {}
    cluster_s = {}
    u1, s1, cluster_vh[0] = np.linalg.svd(dataset_1.T, full_matrices=False)
    u2, s2, cluster_vh[1] = np.linalg.svd(dataset_2.T, full_matrices=False)

    pi = (s1[0] / s2[0]) ** 2
    if pi > 1:
        pi = 1 / pi

    cluster_s[0] = utils.get_energy(s1)
    cluster_s[1] = utils.get_energy(s2)

    gamma_list = []
    for g1 in range(2):
        for g2 in range(2):
            if g1 != g2:
                gamma = np.max(
                    cluster_vh[g1][0] @ (np.diag(cluster_s[g2]) @ cluster_vh[g2]).T
                )
                if verbose:
                    print(f"Group {g1} vs Group {g2}")
                    print(f"maximum Cosine similarity: {gamma}")
                gamma_list.append(gamma)
    return max(gamma_list), pi


##### FOR TESTING #####
def one_leave_out_test(
    dataset, T0, num_sv=None, method="linreg", prefix="fit", target_units=None
):
    warnings.simplefilter(action="ignore", category=FutureWarning)

    # dataset comes in as T by n, so columns are units
    if target_units is None:
        target_units = dataset.columns

    df_result = pd.DataFrame(columns=["id", "rank", "metric", "value"])
    for target in target_units:
        M = Matrix(dataset, T0=T0, target_name=target)
        if num_sv is not None:
            M.denoise(num_sv=num_sv, transform=False)
        syc = SyntheticControl()
        syc.fit(M.pre_donor, M.pre_target, method=method)
        pre_fit = syc.predict_and_mse(M.pre_donor, M.pre_target)
        post_fit = syc.predict_and_mse(M.post_donor, M.post_target)
        train_result = pd.DataFrame(
            {
                "id": [target],
                "rank": [num_sv],
                "metric": "{}_train".format(prefix),
                "value": [pre_fit],
            }
        )
        test_result = pd.DataFrame(
            {
                "id": [target],
                "rank": [num_sv],
                "metric": "{}_test".format(prefix),
                "value": [post_fit],
            }
        )
        df_result = pd.concat(
            [df_result, train_result, test_result], ignore_index=True, axis=0
        )

    return df_result
