import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from scipy import optimize
from matrix import Matrix


class SyntheticControl:
    def __init__(self):
        self.target = None
        self.method = None

        self.model = None
        self.weights = None

    def fit(
        self,
        pre_donor,
        pre_target,
        method: str = "ols",
        lmbda=None,
        fit_intercept=True,
    ):
        self.target = pre_target.columns.values
        self.method = method
        if method == "ols":
            self.model = linear_model.LinearRegression(fit_intercept=fit_intercept)
            self.model.fit(pre_donor, pre_target)
        elif method == "ridge":
            self.model = linear_model.Ridge(alpha=lmbda, fit_intercept=fit_intercept)
            self.model.fit(pre_donor, pre_target)
        elif method == "lasso":
            self.model = linear_model.Lasso(alpha=lmbda, fit_intercept=fit_intercept)
            self.model.fit(pre_donor, pre_target)
        elif method == "simplex":
            # Abadie style optimization
            # parameters are [w1, ..., wn]
            # constraints: w1 + ... + wn = 1, wi >= 0
            # objective: minimize sum((Xw - y)^2)
            self.model = SimplexLinearRegression(fit_intercept=fit_intercept)
            self.model.fit(pre_donor, pre_target)
    
    def predict_transformed_data(self, M):
        # M is a matrix object
        # this function automatically predicts the pre- and post-intervention predictions
        # using the provided donors in M and for a target in M.
        if not self.method:
            raise ValueError("Model not fitted.")
        else:
            pred = self.model.predict(M.donor)
            pred = pd.DataFrame(pred, index=M.donor.index, columns=self.target)
        
        if M.data_is_transformed:
            pred = M.scaler.inverse_transform(pred.T).reshape(-1,1)
        
        return pd.DataFrame(pred, index=M.donor.index, columns=self.target)
        

    def predict(self, donor):
        if not self.method:
            raise ValueError("Model not fitted.")
        else:
            pred = self.model.predict(donor)
        
        return pd.DataFrame(pred, index=donor.index, columns=self.target)

    def score(self, donor, target):
        return self.model.score(donor, target)

    def predict_and_mse(self, donor, target_true):
        target_pred = self.predict(donor)
        return mean_squared_error(target_pred, target_true)


class SimplexLinearRegression:

    """Linreg model with simplex constraints on the coefficients.

    Simplex means coefs sum to 1 and are nonnegative.
    """

    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        """
        Fit the SimplexLinearRegression model according to the given training data.

        Optimize using SLSQP (Sequential Least SQuares Programming).

        Args:
            X: Training matrix, shape (n_samples, n_features).
            y: Target vector, shape (n_samples,).

        Returns:
            self: Returns an instance of self.
        """
        # Convert pandas objects to numpy arrays if necessary
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X = X.values
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values.ravel()  # ensure y is 1D array

        if self.fit_intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])

        def objective(coef):
            return np.sum((y - X @ coef) ** 2)

        # Simplex constraints
        constraints = ({"type": "eq", "fun": lambda x: np.sum(x) - 1},)  # sum to 1
        bounds = [(0, None) for _ in range(X.shape[1])]  # nonnegative

        initial_guess = np.ones(X.shape[1]) / X.shape[1]  # initialize coef
        result = optimize.minimize(
            objective,
            initial_guess,
            method="SLSQP",
            constraints=constraints,
            bounds=bounds,
        )

        if self.fit_intercept:
            self.intercept_ = result.x[0]
            self.coef_ = result.x[1:]
        else:
            self.coef_ = result.x

        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values

        if self.fit_intercept:
            return self.intercept_ + X @ self.coef_
        else:
            return X @ self.coef_

# Test
# X = np.sin(np.random.rand(10,5))
# print("SHAPE:")
# print(np.shape(X))
# print()
# df = pd.DataFrame(X)
# M = Matrix(df, T0 = 5, target_name = 0)

# print('data\n', df)
# print('\npre_target\n', M.pre_target)
# print('\npre_donor\n', M.pre_donor)
# print('\npost_donor\n', M.post_donor)
# print('\npost_target\n', M.post_target)

# syc = SyntheticControl()
# syc.fit(M.pre_donor, M.pre_target)
# print('\nweights:\n', syc.weights)
# print('\npredict\n', syc.predict(M.donor))
# print('\nR^2:\n', syc.score(M.post_donor, M.post_target))
# print('MSE: ', syc.predict_and_mse(M.post_donor, M.post_target))
