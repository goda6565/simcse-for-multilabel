nohup python -m l2v.experiments.run_jscl \
  --output_dir outputs/l2v/jscl/semeval-es \
  --dataset_name semeval-es \
  --per_device_batch_size 4 \
  --learning_rate 1e-5 \
  --max_length 128 &