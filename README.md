# Obstructive Sleep Apnoea Prediction: A Comprehensive Review and Comparative Study

Our studies are built upon a large and unique dataset with 110,000 patients and hundreds of EHR clinical features collected from the French national registry of sleep apnoea, which can reduce data biases. We study a wide range of:

- **Machine Learning methods**: AdaBoostRegressor, BaggingRegressor, CatBoostRegressor, DecisionTreeRegressor, ExtraTreesRegressor, GaussianProcessRegressor, GradientBoostingRegressor, KNeighborsRegressor, LGBMRegressor, LinearSVR, MLPRegressor, NuSVR, MLPRegressor, NuSVR, PassiveAggresiveRegressor, RandomForestRegressor, Ridge, RidgeCV, SGDRegressor, SVR, XGBRegressor.

- **Deep Learning methods**: AutoInt, CEM, CNN, DANet, FTT, GANDALF, GATE, LSTM, Node, RNN, TabNet, TabTransformer.

## How to run?

Run the training script with grid search. Prepare your config for grid search in folder `config`:
``` python
python DL_OSA_Tabular_Tuning.py --model $m --target_col $t --batch_size 1024 --imp median_const --imb $i --uncleaned_data 1
python hyperparameter_tuning_impute.py --model_name LGBMClassifier --target_col Severity

```

### Contact:
Gmail: <theduckieducks@gmail.com>