nohup python -m bert.experiments.run_dscl \
  --output_dir outputs/bert/dscl/wrime \
  --dataset_name wrime \
  --per_device_batch_size 32 \
  --learning_rate 1e-5 \
  --max_length 128 &