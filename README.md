# Obstructive Sleep Apnoea Prediction: A Comprehensive Review and Comparative Study

Our studies are built upon a large and unique dataset with 110,000 patients and hundreds of EHR clinical features collected from the French national registry of sleep apnoea, which can reduce data biases. We study a wide range of:

- **Machine Learning methods**: Neareast Centroid, Hist Gradient Boosting, Gradient Boosting, LGBM, MLP, Bagging, Random Forest, SGD, XGBoost, GaussianNB, AdaBoost, LDA, MultinomialNB, Logistic Regression, CatBoost, Extra Tree, BernoulliNB, QDA, LinearSVC, Decision Tree, Ridge, KNN, Perceptron, Passive Agressive, NuSVC, SVC, Radius Neighbors.

- **Deep Learning methods**: CEM, CNN, DANet, GATE, LSTM, Node, RNN, TabNet, TabTransformer, GRU, GCN, DBN.

## How to run?

Run the training script with grid search. Prepare your config for grid search in folder `config`:
``` python
python DL_OSA_Tabular_Tuning.py --model $m --target_col $t --batch_size 1024 --imp median_const --imb $i --uncleaned_data 1
python hyperparameter_tuning_impute.py --model_name LGBMClassifier --target_col Severity

```

### Contact:
Gmail: <theduckieducks@gmail.com>
