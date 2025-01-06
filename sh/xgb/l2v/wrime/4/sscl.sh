nohup python -m xgb.experiments.main \
    --model_name l2v \
    --model_type sscl \
    --dataset_name wrime \
    --output_dir outputs/l2v/sscl/wrime/4 \
    --max_length 128 &