# -----------------------------IMPORTS--------------------------------------#
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier


# --------------------------------------------------------------------------#


# -----------------------------FUNCTIONS------------------------------------#


def create_new_model():
    return RandomForestClassifier()


def save_feature_importances_to_file(feature_importance, feat_names):
    """Saves the feature importances array to a text file

    Arguments:
        feature_importance {array} -- An array containing relative feature importances
        feat_names {array} -- An array containing relative feature names
        data_type {string} -- The data type
        used_permutation {bool} -- Whether permutation was used for this data
    """
    header = np.reshape(np.asarray(['FeatName', 'RelFeatImp']), (1, 2))
    feat_imp_vector = np.column_stack((feat_names, np.asarray(feature_importance, dtype=str)))
    feat_vector_save = np.vstack((header, feat_imp_vector))

    np.savetxt(f'feat_imp.txt', feat_vector_save, fmt='%s', delimiter='\t')


def save_model_best_params(model_config, shuffle_number, output_path):
    with open(f'{output_path}/shuffle_results/{shuffle_number}/best_model_params.json', 'w') as outfile:
        json.dump(model_config, outfile, indent=4)
        outfile.write("\n")
# ------------------------------------------------------------------------#
