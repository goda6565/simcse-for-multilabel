nohup python -m l2v.experiments.run_jscl \
  --output_dir outputs/l2v/jscl/eurlex/16 \
  --dataset_name eurlex \
  --per_device_batch_size 16 \
  --learning_rate 1e-5 \
  --max_length 512 &