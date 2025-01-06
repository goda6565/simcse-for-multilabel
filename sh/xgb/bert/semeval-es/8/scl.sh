nohup python -m xgb.experiments.main \
    --model_name bert \
    --model_type scl \
    --dataset_name semeval-es \
    --output_dir outputs/bert/scl/semeval-es/8 \
    --max_length 128 &
