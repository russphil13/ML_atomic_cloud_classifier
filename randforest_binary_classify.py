from custom.spe import get_clouds, get_analysis_results
from custom.spe import get_ROI_params
from custom.model_development import classify_function_analysis
from custom.model_development import getstats_fromstream
from custom.model_development import load_datasets, make_datasets
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost.sklearn import XGBRFClassifier

#set types and number of examples from each class in each set
set_types = ['training', 'test']
num_ex = [260, 64]
datasets_dict = dict(zip(set_types, num_ex))

class_names = ['empty', 'cloud']
seed = 52

params_file = Path('/home/bob/development/experiment_data/2018/'
                   'cloudparams_s20.csv')

dir_training_images = ('/home/bob/development/atomic_cloud_training_data/'
                       'training_data')

path_images_left = Path(dir_training_images, 'left_clouds/seed_13')
path_images_right = Path(dir_training_images, 'right_clouds/seed_31')

dir_model_main = ('/home/bob/development/ml_algorithms/atomic_cloud/'
                  'randforest_binary_classify/models')

path_model_id = Path(dir_model_main, f'num_ex_{num_ex[0]}_seed_{seed}')

make_datasets(class_names, datasets_dict, path_images_left, path_model_id, seed)

X_L, X_test_L, y_L, y_test_L = load_datasets(path_model_id,
                                             path_images_left)

_, scale_L, _ = getstats_fromstream(path_model_id, path_images_left)
scale_rnd_L = np.around(scale_L)
X_scl_L = X_L / scale_rnd_L
X_test_scl_L = X_test_L / scale_rnd_L

X_train_scl_L, X_eval_scl_L, y_train_L, y_eval_L = train_test_split(
    X_scl_L, y_L, train_size=0.8, random_state=123)

make_datasets(class_names, datasets_dict, path_images_right, path_model_id, seed)

X_R, X_test_R, y_R, y_test_R = load_datasets(path_model_id,
                                             path_images_right)

_, scale_R, _ = getstats_fromstream(path_model_id, path_images_right)
scale_rnd_R = np.around(scale_R)
X_scl_R = X_R / scale_rnd_R
X_test_scl_R = X_test_R / scale_rnd_R

X_train_scl_R, X_eval_scl_R, y_train_R, y_eval_R = train_test_split(
    X_scl_R, y_R, train_size=0.8, random_state=123)

#Train model and evaluate
_, roiL, roiR = get_ROI_params(params_file)

clouds_L, _ = get_clouds(roiL)
X_clouds_L = clouds_L.reshape((clouds_L.shape[0], -1))
X_clouds_scl_L = X_clouds_L / scale_rnd_L
model_L_params = {'colsample_bytree': 0.07,
                  'gamma': 0.005,
                  'max_depth': 3,
                  'min_child_weight': 3,
                  'n_estimators': 500,
                  'objective': 'binary:logistic',
                  'random_state': 10,
                  'reg_alpha': 9,
                  'reg_lambda': 0,
                  'subsample': 0.6,
                  'verbosity': 0}
model_L = XGBRFClassifier(**model_L_params)
model_L.fit(X_train_scl_L, y_train_L, eval_set=eval_set,
             eval_metric='auc', early_stopping_rounds=20,
             verbose=False)
pred_L = model_L.predict(X_clouds_scl_L)

clouds_R, _ = get_clouds(roiR)
X_clouds_R = clouds_R.reshape((clouds_R.shape[0], -1))
X_clouds_scl_R = X_clouds_R / scale_rnd_R
model_R_params = {'colsample_bytree': 0.07,
                  'gamma': 0.005,
                  'max_depth': 3,
                  'min_child_weight': 3,
                  'n_estimators': 500,
                  'objective': 'binary:logistic',
                  'random_state': 10,
                  'reg_alpha': 9,
                  'reg_lambda': 0,
                  'subsample': 0.6,
                  'verbosity': 0}
model_R = XGBRFClassifier(**model_R_params)
model_R.fit(X_train_scl_R, y_train_R, eval_set=eval_set,
            eval_metric='auc', early_stopping_rounds=20,
            verbose=False)
pred_R = model_R.predict(X_clouds_scl_R)

#Load results from standard analysis and compare to analysis after
#applying a classifier to the left and right clouds in each image.
results_list = np.array(get_analysis_results())

classify_results = classify_function_analysis(pred_L, pred_R)

plt.plot(results_list[:, -2], results_list[:, -1], 'k.',
         classify_results[:, -2], classify_results[:, -1], 'r.')
plt.show()
