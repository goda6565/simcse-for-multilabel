python -m bert.experiments.run_dscl \
  --output_dir outputs/bert/dscl/eurlex/16 \
  --dataset_name eurlex \
  --per_device_batch_size 16 \
  --learning_rate 1e-5 \
  --max_length 512 \