nohup python -m l2v.experiments.run_sscl \
  --output_dir outputs/l2v/sscl/wrime/16 \
  --dataset_name wrime \
  --per_device_batch_size 16 \
  --learning_rate 1e-5 \
  --max_length 128 &