nohup python -m l2v.experiments.run_dscl \
  --output_dir outputs/l2v/dscl/eurlex/8 \
  --dataset_name eurlex \
  --per_device_batch_size 8 \
  --learning_rate 1e-5 \
  --max_length 512 &