python -m l2v.experiments.run_dscl \
  --output_dir outputs/l2v/dscl/eurlex \
  --dataset_name eurlex \
  --per_device_batch_size 32 \
  --learning_rate 1e-5 \
  --record_steps 120 \
  --max_length 512 \