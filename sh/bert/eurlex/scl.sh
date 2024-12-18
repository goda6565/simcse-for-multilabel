python -m bert.experiments.run_scl \
  --output_dir outputs/bert/scl/eurlex \
  --dataset_name eurlex \
  --per_device_batch_size 32 \
  --learning_rate 1e-5 \
  --record_steps 120 \
  --max_length 512 \