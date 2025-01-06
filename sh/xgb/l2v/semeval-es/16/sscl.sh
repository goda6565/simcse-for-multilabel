nohup python -m xgb.experiments.main \
    --model_name l2v \
    --model_type sscl \
    --dataset_name semeval-es \
    --output_dir outputs/l2v/sscl/semeval-es/16 \
    --max_length 128 &