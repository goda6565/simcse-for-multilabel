nohup python -m xgb.experiments.main \
    --model_name bert \
    --model_type sscl \
    --dataset_name semeval-en \
    --output_dir outputs/bert/sscl/semeval-en/8 \
    --max_length 128 &