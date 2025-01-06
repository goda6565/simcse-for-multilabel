python -m bert.experiments.run_scl \
  --output_dir outputs/bert/scl/semeval-es/8 \
  --dataset_name semeval-es \
  --per_device_batch_size 8 \
  --learning_rate 1e-5 \
  --max_length 128 \