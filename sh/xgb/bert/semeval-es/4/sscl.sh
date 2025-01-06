nohup python -m xgb.experiments.main \
    --model_name bert \
    --model_type sscl \
    --dataset_name semeval-es \
    --output_dir outputs/bert/sscl/semeval-es/4 \
    --max_length 128 &