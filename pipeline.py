# Step 5: Implement Machine Model Training & Comparison Pipeline

import pandas as pd
import numpy as np
import smogn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from scipy.ndimage import convolve1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang

def load_data(file_path):
    return pd.read_csv(file_path)

def load_and_augment_data_smogn(file_path, target_column='MSRP'):
    data = load_data(file_path)
    try:
        data_smogn = smogn.smoter(data=data, y=target_column)
    except ValueError:
        print("Oops! Synthetic data contains missing values.")
        data = data.dropna()
        data_smogn = smogn.smoter(data=data, y=target_column)
    return data_smogn

def train_rf_model(X_train, y_train, param_grid):
    rf_model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_rf_model = grid_search.best_estimator_
    return best_rf_model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    def weighted_squared_error(y_true, y_pred, underestimation_weight, overestimation_weight):
        errors = y_true - y_pred
        weighted_errors = np.where(errors < 0, (errors ** 2) * underestimation_weight, (errors ** 2) * overestimation_weight)
        return np.mean(weighted_errors)
    
    underestimation_weight = 1
    overestimation_weight = 0  

    y_test_sorted = np.sort(y_test)
    quantile_idx = int(0.99 * len(y_test_sorted))
    threshold = y_test_sorted[quantile_idx]
    quantile_loss = np.mean(np.maximum(0, y_pred[y_test > threshold] - y_test[y_test > threshold]))

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    wmse = weighted_squared_error(y_test, y_pred, underestimation_weight, overestimation_weight)
    
    return mae, mse, wmse, quantile_loss

def plot_predictions(y_pred, y_test):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, y_test, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Predicted vs Actual')
    plt.grid(True)
    plt.show()

def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))
    
    return kernel_window

def prepare_weights(labels, reweight, lds=True, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
    assert reweight in {'none', 'inverse', 'sqrt_inv'}
    assert reweight != 'none' if lds else True, \
        "Set reweight to \'sqrt_inv\' (default) or \'inverse\' when using LDS"

    value_dict = {x: 0 for x in list(set(labels))}  # initialize value dictionary with labels as keys
    for label in labels:
        value_dict[label] += 1  # increment counts of labels which occur multiple times
    if reweight == 'sqrt_inv':
        value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}
    elif reweight == 'inverse':
        value_dict = {k: np.clip(v, 5, 1000) for k, v in value_dict.items()}  # clip weights for inverse re-weight
    num_per_label = [value_dict[label] for label in labels]
    if not len(num_per_label) or reweight == 'none':
        return None
    print(f"Using re-weighting: [{reweight.upper()}]")
    
    if lds:
        lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
        print(f'Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})')
        # apply kernel to the reweighted values
        smoothed_value = convolve1d(
            np.asarray([v for _, v in value_dict.items()]), weights=lds_kernel_window, mode='constant')
        value_dict_keys = list(value_dict.keys())
        num_per_label = [smoothed_value[value_dict_keys.index(label)] for label in labels]
    
    weights = [np.float32(1 / x) for x in num_per_label]
    scaling = len(weights) / np.sum(weights)
    weights = [scaling * x for x in weights]
    
    return weights

def main():
    data_benchmark = load_data('data_versions/binarized_standardized_data.csv')
    X_benchmark = data_benchmark.drop('MSRP', axis=1)
    y_benchmark = data_benchmark['MSRP']
    X_train_benchmark, X_test_benchmark, y_train_benchmark, y_test_benchmark = train_test_split(X_benchmark, y_benchmark, test_size=0.2, random_state=42)
    benchmark_model = train_rf_model(X_train_benchmark, y_train_benchmark, {'n_estimators': [100]})
    y_pred_benchmark = benchmark_model.predict(X_test_benchmark)
    mae_benchmark, mse_benchmark, wmse_benchmark, quantile_loss_benchmark = evaluate_model(benchmark_model, X_test_benchmark, y_test_benchmark)

    print("Benchmark Model:")
    print("Mean Absolute Error:", mae_benchmark)
    print("Mean Squared Error:", mse_benchmark)
    print("Weighted Mean Squared Error:", wmse_benchmark)
    print("Quantile Loss (for the upper 1% of the price distribution):", quantile_loss_benchmark)

    plot_predictions(y_pred_benchmark, y_test_benchmark)

    data = load_and_augment_data_smogn('data_versions/famd_data.csv')
    X = data.drop('MSRP', axis=1)
    y = data['MSRP']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    weights = prepare_weights(labels=y_train, reweight='inverse', lds=True, lds_kernel='laplace', lds_ks=3, lds_sigma=1)

    weighted_y_train = y_train.values * weights  

    model = train_rf_model(X_train, weighted_y_train, param_grid = {
        'n_estimators': [50, 100, 200, 300, 400], 
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
    })

    y_pred = model.predict(X_test)

    mae, mse, wmse, quantile_loss = evaluate_model(model, X_test, y_test)

    print("Final Model:")
    print("Mean Absolute Error:", mae)
    print("Mean Squared Error:", mse)
    print("Weighted Mean Squared Error:", wmse)
    print("Quantile Loss (for the upper 1% of the price distribution):", quantile_loss)

    plot_predictions(y_pred, y_test)

if __name__ == "__main__":
    main()
