python -m bert.experiments.run_sscl \
  --output_dir outputs/bert/sscl/eurlex/16 \
  --dataset_name eurlex \
  --per_device_batch_size 16 \
  --learning_rate 1e-5 \
  --max_length 512 \