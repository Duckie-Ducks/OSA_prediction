CUDA_VISIBLE_DEVICES=0 python DL_tabular_gridsearch.py --model CNN  --target_col Severity
CUDA_VISIBLE_DEVICES=0 python DL_tabular_gridsearch.py --model LSTM --target_col Severity
CUDA_VISIBLE_DEVICES=0 python DL_tabular_gridsearch.py --model RNN --target_col Severity 
CUDA_VISIBLE_DEVICES=0 python DL_tabular_gridsearch.py --model GRU --target_col Severity
CUDA_VISIBLE_DEVICES=0 python DL_tabular_gridsearch.py --model GCN --target_col Severity

CUDA_VISIBLE_DEVICES=0 python DL_tabular_gridsearch.py --model CNN --target_col AHI_5
CUDA_VISIBLE_DEVICES=0 python DL_tabular_gridsearch.py --model LSTM --target_col AHI_5 
CUDA_VISIBLE_DEVICES=0 python DL_tabular_gridsearch.py --model RNN --target_col AHI_5
CUDA_VISIBLE_DEVICES=0 python DL_tabular_gridsearch.py --model GRU --target_col AHI_5
CUDA_VISIBLE_DEVICES=0 python DL_tabular_gridsearch.py --model GCN --target_col AHI_5
