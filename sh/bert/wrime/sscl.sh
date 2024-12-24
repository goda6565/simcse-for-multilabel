nohup python -m bert.experiments.run_sscl \
  --output_dir outputs/bert/sscl/wrime \
  --dataset_name wrime \
  --per_device_batch_size 32 \
  --learning_rate 1e-5 \
  --max_length 128 &