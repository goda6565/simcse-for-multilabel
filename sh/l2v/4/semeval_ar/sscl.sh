nohup python -m l2v.experiments.run_sscl \
  --output_dir outputs/l2v/sscl/semeval-ar/4 \
  --dataset_name semeval-ar \
  --per_device_batch_size 4 \
  --learning_rate 1e-5 \
  --max_length 128 &