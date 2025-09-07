#!/bin/bash
#features_name = `echo "all_features Demographic Measurements Symptoms Questionnaires Comorbidities" `
#target_col = ["Severity", "AHI_5", "AHI_15", "AHI_30"]

for t in "Severity" "AHI_5" "AHI_15" "AHI_30"
do
    for m in "DANet" "TabTransformer"
    do
        python DL_OSA_Tabular_Eval.py --model $m  --target_col Severity  --imb None --features_name all_features
    done
done