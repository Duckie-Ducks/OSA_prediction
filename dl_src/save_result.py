import numpy as np
import pandas as pd
import os
from pathlib import Path

#model_names = ['GANDALF', 'CEM', 'AutoInt', 'DANet', 'Node','GATE','TabNet','TabTransformer', 'cnn', 'lstm']
model_names = [ 'DANet','TabTransformer']
target_cols = ['AHI'] #, 'AHI_5', 'AHI_15', 'AHI_30']
#feature_names = [ "Demographic", "Measurements", "Symptoms", "Questionnaires", 'Comorbidities']
#feature_names = ["Wu", "Mencar", "Ustun", "Huang", "Rodruiges"]
#feature_names = ['SMOTE', 'ADASYN']
feature_names = ['None']
#imb_names = ['SMOTE', 'ADASYN']
#imb_names = ['Borderline', 'Under']
imb_names = ['median_const']
#headers = ['model', 'accuracy', 'precision', 'recall_macro', 'recall_weighted', 'f1_weighted', 'f1_macro', 'bal_acc', 'g_mean']
headers = ['model', 'MSE', 'RMSE', 'MAPE', 'MAPE_nonzero', 'MAE']
#all_headers = ['accuracy', 'f1_weighted', 'bal_acc', 'g_mean']
all_headers = ['MSE', 'RMSE', 'MAPE', 'MAPE_nonzero', 'MAE']
'''
main_data = {'Name': [],
			'accuracy': [], 
			'f1_weighted' : [], 
			'bal_acc' : [], 
			'g_mean' : []}

'''

main_data = {'Name': [],
			'MSE': [], 
			'RMSE' : [], 
			'MAPE' : [],
			'MAPE_nonzero' : [],
			'MAE' : []}

for model_name in model_names:
	for task_name in target_cols:
		for imb_name in imb_names:
			for feature_name in feature_names:
				df_result = pd.DataFrame(columns=headers)
				#model_path = 'results_imb/{}/{}/{}/{}'.format(task_name, imb_name, feature_name, model_name)
				#model_path = 'results_80k/{}/{}/{}/{}'.format(task_name, imb_name, feature_name, model_name)
				model_path = 'results_80k_regression/{}/{}/{}/{}'.format(task_name, imb_name, feature_name, model_name)
				'''
				output_folder = 'report_results/{}/{}/{}/{}'.format(task_name, imb_name, feature_name, model_name)
				output_path = os.path.join(output_folder, 'result_for_report.csv')
				Path(output_folder).mkdir(parents=True, exist_ok=True)
				'''
				result_path = os.path.join(model_path, 'full_metric_result.csv')
				df = pd.read_csv(result_path, index_col=0)
				#all_headers = list(df)
				row = {'model':model_name}
				
				all_data = {}
				main_data['Name'].append('{}_{}_{}_{}'.format(model_name, task_name, imb_name, feature_name))
				for header in all_headers:
					vals = np.array(df[header][model_name+'test'].values)
					mean_val = round(np.mean(vals), 3)
					row[header] = mean_val
					all_data[header] = mean_val
					main_data[header].append(mean_val)
				print(all_data)
				#df_result = pd.concat([df_result, pd.DataFrame([row])], ignore_index=True)
				df_result = pd.DataFrame([all_data])
				#print(output_path)
				#df_result.to_csv(output_path, index = False)

df_main = pd.DataFrame(main_data)
output_folder = 'report_results_fixed'
output_path = os.path.join(output_folder, 'result_for_report_median.csv')
print(output_path)
Path(output_folder).mkdir(parents=True, exist_ok=True)
df_main.to_csv(output_path, index = True)
