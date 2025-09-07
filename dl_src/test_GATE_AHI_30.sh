#!/bin/bash
#features_name = `echo "all_features Demographic Measurements Symptoms Questionnaires Comorbidities" `
#target_col = ["Severity", "AHI_5", "AHI_15", "AHI_30"]
for i in "1" "2" "3" "4"    
    do
    for m in "DANet" "TabTransformer"
    do
        for t in "median_const" "median_most" "mean_const" "mean_most" "bayesian" "rf" "knn" "knn5" "knn10" "completed_hmisc_impute1" "mice_imp1_maxit20" "mice_imp2_maxit20" "mice_imp3_maxit20" "mice_imp4_maxit20" "mice_imp5_maxit20"
        do
            python DL_OSA_Tabular_Tuning.py --model $m --target_col AHI_30 --batch_size 1024 --imp median_const 
        done
    done
done