nohup python -m l2v.experiments.run_sscl \
  --output_dir outputs/l2v/sscl/semeval-ar/8 \
  --dataset_name semeval-ar \
  --per_device_batch_size 8 \
  --learning_rate 1e-5 \
  --max_length 128 &