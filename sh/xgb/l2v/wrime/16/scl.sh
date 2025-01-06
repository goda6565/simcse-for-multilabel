nohup python -m xgb.experiments.main \
    --model_name l2v \
    --model_type scl \
    --dataset_name wrime \
    --output_dir outputs/l2v/scl/wrime/16 \
    --max_length 128 &
