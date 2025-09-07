#!/bin/bash
cd ../src
# python hyperparameter_tuning.py --model_name NearestCentroid --target_col AHI_30
# python hyperparameter_tuning.py --model_name LogisticRegression --target_col AHI_30
# python hyperparameter_tuning.py --model_name SGDClassifier --target_col AHI_30
# python hyperparameter_tuning.py --model_name Perceptron --target_col AHI_30

# python hyperparameter_tuning.py --model_name DecisionTreeClassifier --target_col AHI_30
# python hyperparameter_tuning.py --model_name BaggingClassifier --target_col AHI_30
# python hyperparameter_tuning.py --model_name ExtraTreesClassifier --target_col AHI_30
# python hyperparameter_tuning.py --model_name HistGradientBoostingClassifier --target_col AHI_30
# python hyperparameter_tuning.py --model_name LinearDiscriminantAnalysis --target_col AHI_30
# python hyperparameter_tuning.py --model_name QuadraticDiscriminantAnalysis --target_col AHI_30

# python hyperparameter_tuning.py --model_name MLPClassifier --target_col AHI_30
# python hyperparameter_tuning.py --model_name GaussianNB --target_col AHI_30
# python hyperparameter_tuning.py --model_name MultinomialNB --target_col AHI_30
# python hyperparameter_tuning.py --model_name BernoulliNB --target_col AHI_30
# python hyperparameter_tuning.py --model_name SVC --target_col AHI_30
# python hyperparameter_tuning.py --model_name LinearSVC --target_col AHI_30
# python hyperparameter_tuning.py --model_name NuSVC --target_col AHI_30
# python hyperparameter_tuning.py --model_name XGBClassifier --target_col AHI_30
# python hyperparameter_tuning.py --model_name RandomForestClassifier --target_col AHI_30
# python hyperparameter_tuning.py --model_name AdaBoostClassifier --target_col AHI_30

python hyperparameter_tuning.py --model_name AdaBoostClassifier --target_col Severity

# python hyperparameter_tuning.py --model_name GradientBoostingClassifier --target_col AHI_30
# python hyperparameter_tuning.py --model_name RadiusNeighborsClassifier --target_col AHI_30
# python hyperparameter_tuning.py --model_name RidgeClassifierCV --target_col AHI_30
# python hyperparameter_tuning.py --model_name RidgeClassifier --target_col AHI_30
# python hyperparameter_tuning.py --model_name PassiveAggressiveClassifier --target_col AHI_30
# python hyperparameter_tuning.py --model_name LGBMClassifier --target_col AHI_30
# python hyperparameter_tuning.py --model_name CatBoostClassifier --target_col AHI_30
# python hyperparameter_tuning.py --model_name KNeighborsClassifier --target_col AHI_30
# python hyperparameter_tuning.py --model_name GaussianProcessClassifier --target_col AHI_30
