import json
import os
from tqdm import tqdm


def generate_param_space(save_folder="../../config/"):
    # Models parameters
    # Fommat: Classname_params
    clfs_params = {
            "MLPClassifier_params": {
                'hidden_layer_sizes': [(50, 30, 10),(50, 30), (50), (30), (10)],
                'activation': ['tanh', 'relu'],
                'solver': ['sgd', 'adam'],
                'alpha': [0.0001, 0.001, 0.01, 0.1],
                'learning_rate': ['adaptive', 'constant', 'invscaling'],
            },
            "KNeighborsClassifier_params": {
                'n_neighbors': (1,10, 1),
                'leaf_size': (20,40,1),
                'p': (1,2),
                'weights': ('uniform', 'distance'),
                'metric': ('minkowski', 'chebyshev'),
            },
            "NearestCentroid_params": {
                'metric': ('euclidean', 'manhattan'),
            },
            "LogisticRegression_params": {
                'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
                'C': [1, 10, 100, 1000],
                'solver' : ['lbfgs','newton-cg','liblinear','sag','saga'],
            },
            "SGDClassifier_params":  {
                'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
                'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                'alpha': [0.0001, 0.001, 0.01, 0.1],
                'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
            },
            "Perceptron_params": {
                'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                'alpha': [0.0001, 0.001, 0.01, 0.1],
            },
            "DecisionTreeClassifier_params": {
                'max_depth': [10, 30, 50, 70, 90, None],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ["gini", "entropy"]
            },
            "GaussianNB_params": {
                'var_smoothing': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15]
            },
            "MultinomialNB_params": {
                'alpha': [0, 1]
            },
            "BernoulliNB_params": {
                'alpha': [0, 1]
            },
            "SVC_params": [
                {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
                {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
            ],
            "LinearSVC_params": {
                'C': [1, 10, 100, 1000],
                'penalty': ['l1', 'l2', 'elasticnet', 'none'],#, 'elasticnet', 'none'
            },
            "NuSVC_params": {
                'gamma': [0.001, 0.0001], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'nu': [0.1, 0.2, 0.3, 0.4]
            },
            "XGBClassifier_params": {
                'n_estimators': [10, 50, 100],
                'learning_rate': [0.001, 0.01, 0.05],
                'booster': ['gbtree', 'gblinear'],
                'reg_alpha': [0.5, 1],
                'reg_lambda': [0.5, 1],
                'base_score': [0.5, 1]
            },
            "RandomForestClassifier_params": {
                'bootstrap': [True, False],
                'max_depth': [10, 30, 50, 70, 90, None],
                'max_features': ['auto', 'sqrt'],
                'min_samples_leaf': [1, 2, 4],
                'min_samples_split': [2, 5, 10],
                'n_estimators': [10, 50, 100],
            },
            "AdaBoostClassifier_params": {
                'n_estimators': [10, 50, 100],
                'learning_rate': [0.001, 0.01, 0.05],
            },
            "GradientBoostingClassifier_params": {
                'n_estimators': [10, 50, 100],
                'learning_rate': [0.001, 0.01, 0.05],
            },
            "BaggingClassifier_params": {
                'n_estimators': [10, 50, 100],
                'max_features': [0.6, 0.7, 0.8, 0.90, 1.0],
                'bootstrap': [True, False],
                'bootstrap_features': [True, False],
                'oob_score': [True, False],
                'warm_start': [True, False]
            },
            "ExtraTreesClassifier_params": {
                'n_estimators': [10, 50, 100],
                'max_depth': [10, 30, 50, 70, 90, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
            },
            "HistGradientBoostingClassifier_params": {
                'learning_rate': [0.001, 0.01, 0.05],
                "max_leaf_nodes": (5, 10, 30),
            },
            "LinearDiscriminantAnalysis_params": {
                'solver': ('lsqr','eigen'),
                'n_components': (1,5,1),
            },
            "QuadraticDiscriminantAnalysis_params": {
                'reg_param': [0.1, 0.2, 0.3, 0.4, 0.5]
            },
            "GaussianProcessClassifier_params": {
                'kernel': [1, 1, 1, 1, 1]
            },
            "RadiusNeighborsClassifier_params": {
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'leaf_size': (20,40,1),
                'p': (1,2),
                'weights': ('uniform', 'distance'),
                'metric': ('minkowski', 'chebyshev'),
            },
            "RidgeClassifierCV_params": {
                'alphas': [0.1, 1, 2, 5 , 10],
                'fit_intercept': [True, False]
            },
            "RidgeClassifier_params": {
                'alpha': [0.1, 1, 2, 5 , 10],
                'fit_intercept': [True, False]
            },
            "PassiveAggressiveClassifier_params": {
                'C': [1, 10, 100, 1000],
                'fit_intercept': [True, False],
            },


            #LightGBM
            "LGBMClassifier_params": {
                'boosting': ['gbdt' ],
                'num_iterations': [  1500, 2000,5000  ],
                'learning_rate':[  0.05, 0.005 ],
                'num_leaves':[ 7, 15, 31  ],
                'max_depth' :[ 10,15,25],
                'min_data_in_leaf':[15,25 ],
                'feature_fraction': [ 0.6, 0.8,  0.9],
                'bagging_fraction': [  0.6, 0.8 ],
                'bagging_freq': [   100, 200, 400  ]
            },
            #Catboost
            "CatboostClassifier_params": {
                'depth': [4,5,6,7,8,9, 10],
                'learning_rate' : [0.01,0.02,0.03,0.04],
                'iterations'    : [10, 20,30,40,50,60,70,80,90, 100]
            }
    }
    
    for key in clfs_params:
        model_name = key.split("_")[0]
        save_path = os.path.join(save_folder, model_name + '.json')

        with open(save_path, 'w') as f:
            json.dump(clfs_params[key], f, indent=2)


if __name__ == '__main__':
    generate_param_space()
