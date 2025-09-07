#!/bin/bash
#features_name = `echo "all_features Demographic Measurements Symptoms Questionnaires Comorbidities" `
#target_col = ["Severity", "AHI_5", "AHI_15", "AHI_30"]
for i in "ADASYN"
    do
    for m in "DANet"
    do
        for t in "Severity" "AHI_5" "AHI_15" "AHI_30"
        do
            python DL_OSA_Tabular_Tuning.py --model $m --target_col $t --batch_size 1024 --imp median_const --imb $i --uncleaned_data 1
	done
    done
done

