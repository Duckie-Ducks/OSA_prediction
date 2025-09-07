python feature_selection_ensemble.py --model_name LGBMClassifier --target_col Severity --balance_strategy None --features_name all_features
python feature_selection_ensemble.py --model_name LinearDiscriminantAnalysis --target_col Severity --balance_strategy None --features_name all_features
python feature_selection_ensemble.py --model_name RandomForestClassifier --target_col Severity --balance_strategy None --features_name all_features
python feature_selection_ensemble.py --model_name HistGradientBoostingClassifier --target_col Severity --balance_strategy None --features_name all_features
