import json

# Feature sets based on literature review

## Wu et al.
wu = ['Age','BMI','Epworth_scale',  'Abdominal_perimeter', 'Cervical_perimeter','Systolic_BP', 'Diastolic_BP']

## Mencar et al.
mencar = ['BMI', 'Sex', 'Epworth_scale']

## Huang et al.
huang = ['Age','Cervical_perimeter', 'Snoring']

## Uston et al. 
ustun = ['Age','BMI', 'Sex','Diabetes', 'High_BP', 'Current_smoker', 'Former_smoker']

## Rodruiges Jr. et al.
rodruiges = ['Age', 'Nocturia', 'BMI', 'Depression_scale', 'Cervical_perimeter', 'Abdominal_perimeter', 'Diastolic_BP', 'Current_smoker', 'Former_smoker' ]

#====================================================

# Feature sets based on feature category
demographic = ['Sex','Age','Current_smoker','Former_smoker','Sedentary']
measurements = ['Height','Weight','Cervical_perimeter',
               'Abdominal_perimeter','Systolic_BP','Diastolic_BP','Maxillofacial_profile','BMI','High_BP']
comorbidities = ['Asthma','Rhinitis','COPD','Respiratory_fail','Myocardial_infarct','Coronary_fail','Arrhythmias','Stroke',
                 'Heart_fail','Arteriopathy','Gastric_reflux','Glaucoma','Diabetes','Hypercholesterolemia','Hypertriglyceridemia',
                 'Hypo(er)thyroidism','Depression','Obesity','Dysmorphology','Restless_Leg_Syndrome']
symptoms=['Snoring','Diurnal_somnolence','Driving_drowsiness','Morning_fatigue','Morning_headache','Memory_problem',
          'Nocturnal_perspiration','Shortness_of_breath_on_exertion','Nocturia','Drowsiness_accident','Near_miss_accident',
          'Respiratory_arrest']
questionnaires = ['Epworth_scale','Pichots_scale','Depression_scale']

#=====================================================

# Feature sets based on stats (only using top 10)
kbest_chi2 = ['Sex', 'Respiratory_arrest','High_BP','Hypercholesterolemia','Former_smoker','Diabetes','Nocturia',
              'Morning_headache','Coronary_fail','Cervical_perimeter']

kbest_fclass = ['Sex', 'Respiratory_arrest','High_BP','Abdominal_perimeter','Weight','BMI','Systolic_BP',
              'Nocturia','Age','Cervical_perimeter']

rf_impurity = ['Age', 'BMI', 'Abdominal_perimeter',  'Weight', 'Cervical_perimeter', 'Height', 'Pichots_scale', 'Epworth_scale',
               'Systolic_BP', 'Depression_scale']

rf_permutation = ['Age', 'Cervical_perimeter', 'Abdominal_perimeter', 'Respiratory_arrest', 
                  'Sex', 'Maxillofacial_profile', 'Diastolic_BP', 'Weight' ,'BMI' ]

correlation = ['Cervical_perimeter', 'Abdominal_perimeter', 'Age', 'Weight', 'Respiratory_arrest', 'BMI', 
               'High_BP', 'Systolic_BP', 'Nocturia', 'Hypercholesterolemia']

kruskall_chi = ['Cervical_perimeter', 'Abdominal_perimeter', 'Age', 'Weight', 'Respiratory_arrest', 'BMI', 
               'High_BP', 'Systolic_BP', 'Sex', 'Diastolic_BP']

cb_clf = ['Age','Cervical_perimeter', 'Respiratory_arrest', 'Epworth_scale', 'Abdominal_perimeter', 'BMI',
          'Diastolic_BP', 'Systolic_BP', 'Depression_scale', 'Sex']

shap_rf = ['Sex', 'Age', 'Former_smoker', 'Height', 'Weight', 'Cervical_perimeter',
           'Abdominal_perimeter', 'Systolic_BP', 'Diastolic_BP', 'BMI']

# shap_cb has the same top 10 features as shap_rf
features = ['Sex', 'Age', 'Current_smoker', 'Former_smoker', 'Sedentary', 'Height', 'Weight', 'Cervical_perimeter', 
            'Abdominal_perimeter', 'Systolic_BP', 'Diastolic_BP', 'Maxillofacial_profile', 'BMI', 'High_BP', 
            'Asthma', 'Rhinitis', 'COPD', 'Respiratory_fail', 'Myocardial_infarct', 'Coronary_fail', 
            'Arrhythmias', 'Stroke', 'Heart_fail', 'Arteriopathy', 'Gastric_reflux', 'Glaucoma', 'Diabetes', 
            'Hypercholesterolemia', 'Hypertriglyceridemia', 'Hypo(er)thyroidism', 'Depression', 'Obesity', 
            'Dysmorphology', 'Restless_Leg_Syndrome', 'Snoring', 'Diurnal_somnolence', 'Driving_drowsiness', 
            'Morning_fatigue', 'Morning_headache', 'Memory_problem', 'Nocturnal_perspiration', 
            'Shortness_of_breath_on_exertion', 'Nocturia', 'Drowsiness_accident', 'Near_miss_accident', 
            'Respiratory_arrest', 'Epworth_scale', 'Pichots_scale', 'Depression_scale']

feature_list = {}

feature_list['all_features']= features
feature_list['Wu'] = wu
feature_list['Mencar'] = mencar
feature_list['Ustun'] = ustun
feature_list['Huang'] = huang
feature_list['Rodruiges'] = rodruiges
feature_list['Demographic'] = demographic
feature_list['Measurements'] = measurements
feature_list['Comorbidities'] = comorbidities
feature_list['Symptoms'] = symptoms
feature_list['Questionnaires'] = questionnaires
feature_list['KBest Chi2'] = kbest_chi2
feature_list['RF Impurity'] = rf_impurity
feature_list['RF Permutation'] = rf_permutation
feature_list['KBest Fclass'] = kbest_chi2
feature_list['Correlation'] = correlation
feature_list['Kruskall Chi'] = kruskall_chi
feature_list['CatBoost'] = cb_clf
feature_list['SHAP_RF'] = shap_rf


with open("../../feature_config/feature_config.json", 'w') as f:
    json.dump(feature_list, f,)
