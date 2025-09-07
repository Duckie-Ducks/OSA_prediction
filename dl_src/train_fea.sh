#!/bin/bash
#features_name = `echo "all_features Demographic Measurements Symptoms Questionnaires Comorbidities" `
#target_col = ["Severity", "AHI_5", "AHI_15", "AHI_30"]

for t in "Severity" "AHI_5" "AHI_15" "AHI_30"
do
    for f in "Demographic" "Measurements" "Symptoms" "Questionnaires" "Comorbidities"
    do
        #python DL_OSA_Imb.py --model TabTransformer --batch_size 1024 --target_col $t --imb None --features_name $f
        python DL_OSA_Tabular_Eval.py --model DANet --target_col $t --imb None --features_name $f
    done
done
