nohup python -m xgb.experiments.main \
    --model_name bert \
    --model_type scl \
    --dataset_name wrime \
    --output_dir outputs/bert/scl/wrime/16 \
    --max_length 128 &
