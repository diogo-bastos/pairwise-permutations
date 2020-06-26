# -----------------------------IMPORTS-------------------------------#
import numpy as np
from joblib import Parallel, delayed, parallel_backend
from utils import save_feature_importances_to_file, create_new_model
import multiprocessing


# ------------------------------------------------------------------#

def calculate_feature_importances(df_input, df_corr, output_data, n_feature_importance_runs, corr_threshold,
                                  baseline, best_model, train_test_split, metric_func,
                                  train_test_ratio):
    print(f'Calculating feature importance with a threshold of {corr_threshold}.')
    print(f'Number of permutations to calculate for this threshold: {(df_corr > corr_threshold).sum().sum()}')

    num_cores = multiprocessing.cpu_count()
    model_params = best_model.get_params()
    permutations_to_compute = []

    scores_perm = [[None for j in range(0, df_input.shape[1])] for i in range(0, df_input.shape[1])]
    scores_perm_avg = [[None for j in range(0, df_input.shape[1])] for i in range(0, df_input.shape[1])]
    permutation_importances = [None for j in range(0, df_input.shape[1])]

    train_data_ratio = 1 - train_test_ratio
    for index, row in df_corr.iterrows():
        for item in row.iteritems():
            if np.abs(item[1]) > corr_threshold:
                permutations_to_compute.append((index, item))

    for i in np.arange(0, n_feature_importance_runs):
        print(f'permutation run number: {i + 1}')
        # Permute the variable pairs here
        permutation_sequences = []

        for index in range(len(permutations_to_compute)):
            permutation_sequences.append(np.random.permutation(int(df_input.shape[0] * train_data_ratio)))

        with parallel_backend('threading'):
            Parallel(n_jobs=num_cores)(
                delayed(calculate_feature_permutations)(output_data, permutations_to_compute[index][1],
                                                        permutations_to_compute[index][0], df_input, scores_perm,
                                                        corr_threshold, model_params, train_test_split[0],
                                                        train_test_split[1],
                                                        metric_func,
                                                        permutation_sequences[index])
                for index in range(len(permutations_to_compute)))

    # Average the roc auc scores values
    for index, vector in enumerate(scores_perm):
        for index2, vector2 in enumerate(scores_perm[index]):
            if scores_perm[index][index2] is not None and len(scores_perm[index][index2]) > 0:
                scores_perm_avg[index][index2] = np.mean(scores_perm[index][index2])

    for index, vector in enumerate(scores_perm_avg):

        permutation_importances[index] = 0
        denominator = 0

        for index2, vector2 in enumerate(vector):
            if scores_perm_avg[index][index2] is not None:
                permutation_importances[index] += \
                    np.abs(df_corr[df_input.columns[index]][index2]) * (
                            baseline - scores_perm_avg[index][index2])/2
                denominator += np.abs(df_corr[df_input.columns[index]][index2])

        if denominator > 1:
            permutation_importances[index] /= denominator

    feature_importance = np.round(100.0 * (permutation_importances / np.nanmax(permutation_importances)), 2)

    sorted_idx = np.argsort(feature_importance)

    feat_names_for_plot = df_input.columns
    save_feature_importances_to_file(feature_importance[sorted_idx], np.array(feat_names_for_plot)[sorted_idx])

    return feat_names_for_plot, sorted_idx


def calculate_feature_permutations(output_data, item, index, df_input, scores_perm, corr_threshold, model_params,
                                   train_idx, test_idx, metric_func, perm):
    result = None

    feat1 = index
    feat2 = item[0]

    matrix_index1 = np.where(df_input.columns == feat1)[0][0]
    matrix_index2 = np.where(df_input.columns == feat2)[0][0]

    model = create_new_model()
    model.set_params(**model_params)
    X = df_input.to_numpy()
    y = output_data
    # if there is a correlation and it's not the same variable correlating with itself
    if np.abs(item[1]) > corr_threshold and index != item[0]:

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        save1 = X_train[:, matrix_index1].copy()
        save2 = X_train[:, matrix_index2].copy()

        X_train[:, matrix_index1] = X_train[perm, matrix_index1]
        X_train[:, matrix_index2] = X_train[perm, matrix_index2]

        y_pred = model.fit(X_train, y_train).predict(X_test)

        result = metric_func(y_test, y_pred)

        X_train[:, matrix_index1] = save1
        X_train[:, matrix_index2] = save2

    # Calculate the single variable permutations as well
    elif index == item[0]:

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        save1 = X_train[:, matrix_index1].copy()

        X_train[:, matrix_index1] = X_train[perm, matrix_index1]

        y_pred = model.fit(X_train, y_train).predict(X_test)
        X_train[:, matrix_index1] = save1

        result = metric_func(y_test, y_pred)

    # Under the correlation threshold
    else:
        return

    if scores_perm[matrix_index1][matrix_index2] is None:
        scores_perm[matrix_index1][matrix_index2] = []

    scores_perm[matrix_index1][matrix_index2].append(result)
