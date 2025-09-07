# Note:

Step to run this repo:

1. Run the training script with grid search
Prepare your config for grid search in folder `config`
Then run, for example:
`python DL_OSA_Tabular_Tuning.py --model GANDALF --batch_size 1024 --target_col Severity`

Depend on the task, you should choose the appropriate target_col ('Severity' for multi-class, 'AHI_5' for binary with cutoff at 5)

Model information: https://pytorch-tabular.readthedocs.io/en/latest/models

2. Run the evaluation:

Example:
`python DL_OSA_Tabular_Eval.py --model GANDALF --target_col Severity`

3. (Optional task): Explain the prediction
TODO: Run LIME to explain the prediction of trained model
