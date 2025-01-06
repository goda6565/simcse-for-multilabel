nohup python -m xgb.experiments.main \
    --model_name l2v \
    --model_type dscl \
    --dataset_name wrime \
    --output_dir outputs/l2v/dscl/wrime/8 \
    --max_length 128 &