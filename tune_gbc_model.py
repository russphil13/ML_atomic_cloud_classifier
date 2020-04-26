from custom.model_development import load_datasets, make_datasets
from custom.model_development import sort_metric_results
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, make_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

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
                  'gbc_binary_classify/models')

path_model_id = Path(dir_model_main, f'num_ex_{num_ex[0]}_seed_{seed}')

#Tune using training set of left-side clouds.
make_datasets(class_names, datasets_dict, path_images_left, path_model_id, seed)

X_L, X_test_L, y_L, y_test_L = load_datasets(path_model_id,
                                             path_images_left)

_, scale_L, _ = getstats_fromstream(path_model_id, path_images_left)
scale_rnd_L = np.around(scale_L)
X_scl_L = X_L / scale_rnd_L
X_test_scl_L = X_test_L / scale_rnd_L

X_train_scl_L, X_eval_scl_L, y_train_L, y_eval_L = train_test_split(
    X_scl_L, y_L, train_size=0.8, random_state=123)

#Tune using training set of right-side clouds.
#make_datasets(class_names, datasets_dict, path_images_right, path_model_id, seed)
#
#X_R, X_test_R, y_train_R, y_R = load_datasets(path_model_id,
#                                              path_images_right)
#
#_, scale_R, _ = getstats_fromstream(path_model_id, path_images_right)
#scale_rnd_R = np.around(scale_R)
#X_scl_R = X_R / scale_rnd_R
#X_test_scl_R = X_test_R / scale_rnd_R

early_stopping_rounds = 20
eval_set = [(X_eval_scl_L, y_eval_L)]
fixed_params = {'n_estimators': 500,
                'random_state': 10,
                'verbosity': 0}

#Tuning stages.

#Stage 1 - Search for learning_rate
#var_params = {'colsample_bytree': 0.15,
#              'gamma': 0,
#              'max_depth': 10,
#              'min_child_weight': 3,
#              'reg_alpha': 0,
#              'reg_lambda': 0,
#              'subsample': 1}
#learning_rates = [0.0001, 0.001, 0.01, 0.1, 1]
#params_grid = [{'learning_rate': learning_rates}]

#Stage 2 - Search for max_depth and min_child_weight.
#var_params = {'colsample_bytree': 0.15,
#              'gamma': 0,
#              'learning_rate': 0.001,
#              'reg_alpha': 0,
#              'reg_lambda': 0,
#              'subsample': 1}
##max_depth = [3, 7, 11, 15, 19, 24]
##min_child_weight = [1, 3, 5, 7, 9]
#max_depth = [5, 20, 50, 75, 100]
#min_child_weight = [3, 10, 20, 30]
#params_grid = [{'max_depth': max_depth,
#                'min_child_weight': min_child_weight}]
#param_dists = params_grid
#n_iter = 18

#Stage 3 - Search for colsample_bytree and subsample.
#var_params = {'gamma': 0,
#              'learning_rate': 0.001,
#              'max_depth': 9,
#              'min_child_weight': 5,
#              'reg_alpha': 0,
#              'reg_lambda': 0}
#colsample_bytree = [0.02, 0.05, 0.1, 0.2, 0.5]
#colsample_bytree = [i/10 + 0.05 for i in range(1, 5)]
#subsample = [i / 10 for i in range(2, 11, 2)]
#params_grid = [{'colsample_bytree': colsample_bytree,
#                'subsample': subsample}]
#param_dists = params_grid
#n_iter = 25

#Stage 4 - Search for gamma, reg_alpha and reg_lambda.
#var_params = {'colsample_bytree': 0.35,
#              'learning_rate': 0.001,
#              'max_depth': 9,
#              'min_child_weight': 5,
#              'subsample': 0.4}
#gamma = [0.001, 0.01, 0.1, 1, 10]
#reg_alpha = [0.001, 0.01, 0.1, 1, 10]
#reg_lambda = [0.001, 0.01, 0.1, 1, 10]
#gamma = [0.01, 0.1, 1]
#reg_alpha = [10, 100, 1000]
#reg_lambda = [10, 100, 1000]
#params_grid = [{'gamma': gamma, 'reg_alpha': reg_alpha, 'reg_lambda': [0]},
#               {'gamma': gamma, 'reg_alpha': [0], 'reg_lambda': reg_lambda}]
#gamma = [0.1]
#reg_lambda = [i * 10 for i in range(1, 11)]
#params_grid = [{'gamma': gamma, 'reg_alpha': [0], 'reg_lambda': reg_lambda}]
#param_dists = params_grid
#n_iter = 25

#Stage 5 - Check learning_rate again.
var_params = {'colsample_bytree': 0.35,
              'gamma': 0.1,
              'max_depth': 9,
              'min_child_weight': 5,
              'reg_alpha': 0,
              'reg_lambda': 100,
              'subsample': 0.4}
learning_rates = [i / 10000 for i in range(5, 16)]
params_grid = [{'learning_rate': learning_rates}]

estimator = XGBClassifier(**fixed_params, **var_params)

crossval = RepeatedStratifiedKFold(n_splits=6, n_repeats=3, random_state=3)

my_prec_scorer = make_scorer(precision_score, pos_label=class_names[0])
my_recall_scorer = make_scorer(recall_score, pos_label=class_names[0])

metrics = {'accuracy': make_scorer(accuracy_score),
           'precision': my_prec_scorer,
           'recall': my_recall_scorer}

print(f'# Tuning hyper-parameters')
print()

model_search = GridSearchCV(estimator, params_grid, cv=crossval,
                            scoring=metrics, refit=False)

#model_search = RandomizedSearchCV(estimator, param_dists,
#                                  n_iter=n_iter, cv=crossval,
#                                  scoring=metrics, refit=False)

#Fit model with left-side clouds.
model_search.fit(X_train_scl_L, y_train_L, eval_set=eval_set,
                 eval_metric='auc', early_stopping_rounds=20,
                 verbose=False)

#Fit model with right-side clouds.
#model_search.fit(X_train_scl_R, y_train_R, eval_set=eval_set,
#                 eval_metric='auc', early_stopping_rounds=20,
#                 verbose=False)

print("Grid scores by metric on development set:")
print()

results = model_search.cv_results_
means = (results[f'mean_test_{key}'] for key in metrics.keys())
stds = (results[f'std_test_{key}'] for key in metrics.keys())

metrics_results_list = ([m ,s, results['params']]
                        for m, s in zip(means, stds))

metrics_params_results = {k:v for k, v
                          in zip(metrics.keys(), metrics_results_list)}

for key, value in metrics_params_results.items():
    results_sort_desc = sort_metric_results(value)
    print(f'Results for {key} metric:')
    print()
    for _, params, std, mean in results_sort_desc:
        print(f"{mean:0.3f} +/- {std:0.3f} for {params!r}")
    print()
   
# Run this to check performance on the test set.
#model_L_params = {'colsample_bytree': 0.35,
#                  'gamma': 0.1,
#                  'learning_rate': 0.001,
#                  'max_depth': 9,
#                  'min_child_weight': 5,
#                  'n_estimators': 500,
#                  'random_state': 10,
#                  'reg_alpha': 0,
#                  'reg_lambda': 100,
#                  'subsample': 0.4,
#                  'random_state': 10,
#                  'verbosity': 0}
#model_L = XGBClassifier(**model_L_params)
#model_L.fit(X_train_scl_L, y_train_L, eval_set=eval_set,
#            eval_metric='auc', early_stopping_rounds=20,
#            verbose=False)
#print("Detailed classification report:")
#print()
#y_true, y_pred = y_test_L, model_search.predict(X_test_scl_L)
#print(classification_report(y_true, y_pred))
#print()
