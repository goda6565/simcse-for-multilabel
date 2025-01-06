python -m bert.experiments.run_scl \
  --output_dir outputs/bert/scl/wrime/4 \
  --dataset_name wrime \
  --per_device_batch_size 4 \
  --learning_rate 1e-5 \
  --max_length 128 \