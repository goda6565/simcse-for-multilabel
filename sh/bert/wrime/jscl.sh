python -m bert.experiments.run_jscl \
  --output_dir outputs/bert/jscl/wrime \
  --dataset_name wrime \
  --per_device_batch_size 32 \
  --learning_rate 1e-5 \
  --record_steps 120 \
  --max_length 128 \