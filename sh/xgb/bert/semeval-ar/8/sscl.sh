nohup python -m xgb.experiments.main \
    --model_name bert \
    --model_type sscl \
    --dataset_name semeval-ar \
    --output_dir outputs/bert/sscl/semeval-ar/8 \
    --max_length 128 &