nohup python -m l2v.experiments.run_jscl \
  --output_dir outputs/l2v/jscl/semeval-en/8 \
  --dataset_name semeval-en \
  --per_device_batch_size 8 \
  --learning_rate 1e-5 \
  --max_length 128 &