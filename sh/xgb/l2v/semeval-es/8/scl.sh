nohup python -m xgb.experiments.main \
    --model_name l2v \
    --model_type scl \
    --dataset_name semeval-es \
    --output_dir outputs/l2v/scl/semeval-es/8 \
    --max_length 128 &
