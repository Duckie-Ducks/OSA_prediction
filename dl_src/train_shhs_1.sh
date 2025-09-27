CUDA_VISIBLE_DEVICES=0 python DL_OSA_Tabular_hssh_1.py --model GANDALF --target_col Severity  --batch_size 32
CUDA_VISIBLE_DEVICES=0 python DL_OSA_Tabular_hssh_1.py --model TabNet --target_col Severity  --batch_size 32
CUDA_VISIBLE_DEVICES=0 python DL_OSA_Tabular_hssh_1.py --model TabTransformer --target_col Severity  --batch_size 32
CUDA_VISIBLE_DEVICES=0 python DL_OSA_Tabular_hssh_1.py --model CEM --target_col Severity  --batch_size 32
CUDA_VISIBLE_DEVICES=0 python DL_OSA_Tabular_hssh_1.py --model GATE --target_col Severity  --batch_size 32
CUDA_VISIBLE_DEVICES=0 python DL_OSA_Tabular_hssh_1.py --model Node --target_col Severity  --batch_size 32
CUDA_VISIBLE_DEVICES=0 python DL_OSA_Tabular_hssh_1.py --model DANet --target_col Severity  --batch_size 32

CUDA_VISIBLE_DEVICES=0 python DL_OSA_Tabular_hssh_1.py --model GANDALF --target_col AHI_5  --batch_size 32
CUDA_VISIBLE_DEVICES=0 python DL_OSA_Tabular_hssh_1.py --model TabNet --target_col AHI_5  --batch_size 32
CUDA_VISIBLE_DEVICES=0 python DL_OSA_Tabular_hssh_1.py --model TabTransformer --target_col AHI_5  --batch_size 32
CUDA_VISIBLE_DEVICES=0 python DL_OSA_Tabular_hssh_1.py --model CEM --target_col AHI_5  --batch_size 32
CUDA_VISIBLE_DEVICES=0 python DL_OSA_Tabular_hssh_1.py --model GATE --target_col AHI_5  --batch_size 32
CUDA_VISIBLE_DEVICES=0 python DL_OSA_Tabular_hssh_1.py --model Node --target_col AHI_5  --batch_size 32
CUDA_VISIBLE_DEVICES=0 python DL_OSA_Tabular_hssh_1.py --model DANet --target_col AHI_5  --batch_size 32