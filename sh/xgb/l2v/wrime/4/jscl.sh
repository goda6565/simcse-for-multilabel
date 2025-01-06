nohup python -m xgb.experiments.main \
    --model_name l2v \
    --model_type jscl \
    --dataset_name wrime \
    --output_dir outputs/l2v/jscl/wrime/4 \
    --max_length 128 &