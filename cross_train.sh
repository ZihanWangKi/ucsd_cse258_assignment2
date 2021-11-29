gpu=$1
dataset_train=$2
dataset_eval=$3

CUDA_VISIBLE_DEVICES=${gpu} python transformers/examples/pytorch/text-classification/run_glue.py \
  --model_name_or_path bert-base-cased \
  --train_file data/${dataset_train}/train.json \
  --validation_file data/${dataset_eval}/eval.json \
  --output_dir models/${dataset_train}-${dataset_eval}.train \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --do_train \
  --do_eval \
  --save_steps 100000000 \
  --evaluation_strategy steps \
  --eval_steps 300