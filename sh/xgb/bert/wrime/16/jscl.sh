nohup python -m xgb.experiments.main \
    --model_name bert \
    --model_type jscl \
    --dataset_name wrime \
    --output_dir outputs/bert/jscl/wrime/16 \
    --max_length 128 &