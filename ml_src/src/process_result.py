import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


def result_process(strategy="AHI_5", balance_strategy="None", feature_name='SHAP_RF'):
    dir_name="/home/ndoan01/OSA_ML/results/" + strategy
    final_metrics = None

    for model_name in os.listdir(dir_name):
        metric_path = os.path.join(dir_name, model_name, model_name+'_'+balance_strategy+"_"+feature_name+'.csv')
        try:
            metrics = pd.read_csv(metric_path)
        except:
            continue
        
        temp_metrics = metrics.groupby(["model", "is_train"]).mean(numeric_only=True)
        temp_metrics = temp_metrics.reset_index()
        temp_metrics = temp_metrics[temp_metrics["is_train"] == False]
        temp_metrics = temp_metrics.drop("is_train", axis=1)
        temp_metrics = temp_metrics.drop("fold", axis=1)

        # To match the summary file
        temp_metrics = temp_metrics.drop("recall_macro", axis=1)
        temp_metrics = temp_metrics.drop("f1_macro", axis=1)
        temp_metrics = temp_metrics.drop("precision_macro", axis=1)
        
        if final_metrics is None:
            final_metrics = temp_metrics
        else:
            final_metrics = pd.concat([final_metrics, temp_metrics])

    final_metrics = final_metrics.reset_index(drop=True)

    fn = dir_name.split("/")[-1]
    final_metrics.to_csv(f"result_{strategy}_{balance_strategy}_{feature_name}.csv")
    print(final_metrics)


def summarize(result_folder="../results/Severity"):
    for model_name in os.listdir(result_folder):
        try:
            if 'json' in model_name or "csv" in model_name:
                continue
            result_path = os.path.join(result_folder, model_name, model_name + "_metrics.csv")
            result = pd.read_csv(result_path)

            result = result[result["is_train"]==False]
            print(model_name + ": ")
            result = result.mean(numeric_only=True)

            print(result)
        except:
            print(model_name + " has not done")


if __name__ == '__main__':
    for feature_name in ['Wu', 'Mencar', 'Huang', 'Ustun', 'Rodruiges']:
        result_process(strategy="AHI_30", feature_name=feature_name)
        result_process(strategy="AHI_15", feature_name=feature_name)
        result_process(strategy="AHI_5", feature_name=feature_name)
        result_process(strategy="Severity", feature_name=feature_name)
