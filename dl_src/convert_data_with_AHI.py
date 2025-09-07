import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data_clean = pd.read_csv("data/OSA_classifier.csv")
print(data_clean.shape)
data = pd.read_csv("data/OFSP_English_rpts_rmv.csv", encoding= 'unicode_escape')
demographic = ['PatientID','Sex','Age','Current_smoker','Former_smoker','Sedentary']
measurements = ['Height','Weight','Cervical_perimeter',
               'Abdominal_perimeter','Systolic_BP','Diastolic_BP','Bloodsugar_level','Blood_total_cholesterol','HDL_level',
               'LDL_level','Triglycerides_level','Maxillofacial_profile','BMI','High_BP','PH']
comorbidities = ['Asthma','Rhinitis','COPD','Respiratory_fail','Myocardial_infarct','Coronary_fail','Arrhythmias','Stroke',
                 'Heart_fail','Arteriopathy','Gastric_reflux','Glaucoma','Diabetes','Hypercholesterolemia','Hypertriglyceridemia',
                 'Hypo(er)thyroidism','Depression','Obesity','Dysmorphology','Restless_Leg_Syndrome','Aerophagia']
demographic_df = data[demographic]
measurements_df = data[measurements]
comorbidities_df = data[comorbidities]

symptoms=['Snoring','Diurnal_somnolence','Driving_drowsiness','Morning_fatigue','Morning_headache','Memory_problem',
          'Nocturnal_perspiration','Shortness_of_breath_on_exertion','Nocturia','Drowsiness_accident','Near_miss_accident',
          'Respiratory_arrest','Skin_lesions']
symptoms_df = data[symptoms]
questionnaires = ['Epworth_scale','Pichots_scale','Depression_scale']
questionnaires_df = data[questionnaires]
osa_simple = [demographic_df,measurements_df,comorbidities_df, symptoms_df, questionnaires_df, data['Apnea_hypopnea_index']] # this is a list
osa_simple_df = pd.concat(osa_simple, axis=1)
osa_simple_df.drop_duplicates(subset ="PatientID", keep = 'first', inplace = True) # this drops repeated Patient ID rows and only keeps the 1st instance
threshold = 80.0
min_count =  int(((100-threshold)/100)*osa_simple_df.shape[0] + 1)  # this calculates the minimum number of samples per column that must have NaN value to be dropped


osa_simple_nadrop = osa_simple_df.dropna( axis=1,thresh=min_count) # drop columns with more than 80% NaN
osa_simple_nadrop.dropna(subset=['Apnea_hypopnea_index'], inplace = True)
print(osa_simple_nadrop.shape)
#Check if value is matched
'''
for colname in data_clean.columns[:-1]:
    all_equal = data_clean[colname].equals(data[colname])
    if all_equal == False:
        print('Nope')
'''

print(osa_simple_nadrop)
data_clean['AHI'] = osa_simple_nadrop['Apnea_hypopnea_index'].reset_index(drop=True)
print(data_clean['AHI'])
data_clean.to_csv("data/OSA_classifier_AHI.csv")

complete_patients_binary = osa_simple_nadrop[~osa_simple_nadrop.isnull().any(axis=1)]
print(complete_patients_binary.shape)

data_complete = pd.read_csv("data/OSA_complete_patients.csv")
print(data_complete.shape)

data_complete['AHI'] = complete_patients_binary['Apnea_hypopnea_index'].reset_index(drop=True)
print(complete_patients_binary['Apnea_hypopnea_index'])
data_complete.to_csv("data/OSA_complete_AHI.csv")
