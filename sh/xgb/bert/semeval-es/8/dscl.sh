nohup python -m xgb.experiments.main \
    --model_name bert \
    --model_type dscl \
    --dataset_name semeval-es \
    --output_dir outputs/bert/dscl/semeval-es/8 \
    --max_length 128 &