#!/bin/bash
cd ../src

# Mencar
python feature_selection.py --model_name SVC --target_col AHI_5 --balance_strategy smote --features_name Mencar
python feature_selection.py --model_name SVC --target_col AHI_15 --balance_strategy smote --features_name Mencar
python feature_selection.py --model_name SVC --target_col AHI_30 --balance_strategy smote --features_name Mencar
python feature_selection.py --model_name SVC --target_col Severity --balance_strategy smote --features_name Mencar

python feature_selection.py --model_name RandomForestClassifier --target_col AHI_5 --balance_strategy smote --features_name Mencar
python feature_selection.py --model_name RandomForestClassifier --target_col AHI_15 --balance_strategy smote --features_name Mencar
python feature_selection.py --model_name RandomForestClassifier --target_col AHI_30 --balance_strategy smote --features_name Mencar
python feature_selection.py --model_name RandomForestClassifier --target_col Severity --balance_strategy smote --features_name Mencar

# Ustun
# python feature_selection.py --model_name RandomForestClassifier --target_col AHI_5 --balance_strategy adasyn --features_name Ustun
# python feature_selection.py --model_name RandomForestClassifier --target_col AHI_15 --balance_strategy adasyn --features_name Ustun
# python feature_selection.py --model_name RandomForestClassifier --target_col AHI_30 --balance_strategy adasyn --features_name Ustun
# python feature_selection.py --model_name RandomForestClassifier --target_col Severity --balance_strategy adasyn --features_name Ustun

# Huang
python feature_selection.py --model_name SVC --target_col AHI_5 --balance_strategy smote --features_name Huang
python feature_selection.py --model_name SVC --target_col AHI_15 --balance_strategy smote --features_name Huang
python feature_selection.py --model_name SVC --target_col AHI_30 --balance_strategy smote --features_name Huang
python feature_selection.py --model_name SVC --target_col Severity --balance_strategy smote --features_name Huang

# Rodrulges
python feature_selection.py --model_name ExtraTreesClassifier --target_col AHI_5 --balance_strategy smote --features_name Rodruiges
python feature_selection.py --model_name ExtraTreesClassifier --target_col AHI_15 --balance_strategy smote --features_name Rodruiges
python feature_selection.py --model_name ExtraTreesClassifier --target_col AHI_30 --balance_strategy smote --features_name Rodruiges
python feature_selection.py --model_name ExtraTreesClassifier --target_col Severity --balance_strategy smote --features_name Rodruiges