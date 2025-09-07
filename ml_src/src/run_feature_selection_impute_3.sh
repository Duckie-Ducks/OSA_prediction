
python feature_selection_impute.py --model_name RandomForestClassifier --target_col Severity --balance_strategy None --features_name all_features
python feature_selection_impute.py --model_name RandomForestClassifier --target_col AHI_5 --balance_strategy None --features_name all_features
python feature_selection_impute.py --model_name RandomForestClassifier --target_col AHI_15 --balance_strategy None --features_name all_features
python feature_selection_impute.py --model_name RandomForestClassifier --target_col AHI_30 --balance_strategy None --features_name all_features

python feature_selection_impute.py --model_name RandomForestClassifier --target_col Severity --balance_strategy smote --features_name all_features
python feature_selection_impute.py --model_name RandomForestClassifier --target_col AHI_5 --balance_strategy smote --features_name all_features
python feature_selection_impute.py --model_name RandomForestClassifier --target_col AHI_15 --balance_strategy smote --features_name all_features
python feature_selection_impute.py --model_name RandomForestClassifier --target_col AHI_30 --balance_strategy smote --features_name all_features

python feature_selection_impute.py --model_name RandomForestClassifier --target_col Severity --balance_strategy adasyn --features_name all_features
python feature_selection_impute.py --model_name RandomForestClassifier --target_col AHI_5 --balance_strategy adasyn --features_name all_features
python feature_selection_impute.py --model_name RandomForestClassifier --target_col AHI_15 --balance_strategy adasyn --features_name all_features
python feature_selection_impute.py --model_name RandomForestClassifier --target_col AHI_30 --balance_strategy adasyn --features_name all_features

python feature_selection_impute.py --model_name RandomForestClassifier --target_col Severity --balance_strategy underline --features_name all_features
python feature_selection_impute.py --model_name RandomForestClassifier --target_col AHI_5 --balance_strategy underline --features_name all_features
python feature_selection_impute.py --model_name RandomForestClassifier --target_col AHI_15 --balance_strategy underline --features_name all_features
python feature_selection_impute.py --model_name RandomForestClassifier --target_col AHI_30 --balance_strategy underline --features_name all_features

python feature_selection_impute.py --model_name RandomForestClassifier --target_col Severity --balance_strategy borderline --features_name all_features
python feature_selection_impute.py --model_name RandomForestClassifier --target_col AHI_5 --balance_strategy borderline --features_name all_features
python feature_selection_impute.py --model_name RandomForestClassifier --target_col AHI_15 --balance_strategy borderline --features_name all_features
python feature_selection_impute.py --model_name RandomForestClassifier --target_col AHI_30 --balance_strategy borderline --features_name all_features

python feature_selection_impute.py --model_name RandomForestClassifier --target_col Severity --balance_strategy None --features_name all_features
python feature_selection_impute.py --model_name RandomForestClassifier --target_col AHI_5 --balance_strategy None --features_name all_features
python feature_selection_impute.py --model_name RandomForestClassifier --target_col AHI_15 --balance_strategy None --features_name all_features
python feature_selection_impute.py --model_name RandomForestClassifier --target_col AHI_30 --balance_strategy None --features_name all_features
