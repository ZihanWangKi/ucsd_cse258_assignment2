gpu=$1
dataset1=$2
dataset2=$3
f=$4

CUDA_VISIBLE_DEVICES=${gpu} python transformers/examples/pytorch/text-classification/run_glue.py \
    --model_name_or_path bert-base-cased \
    --train_file data/${dataset2}/train.json \
    --validation_file data/${dataset1}/eval.json \
    --output_dir models_128/order/start_${dataset2} \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --do_train \
    --do_eval \
    --save_steps 100000000 \
    --evaluation_strategy steps \
    --eval_steps 300

CUDA_VISIBLE_DEVICES=${gpu} python transformers/examples/pytorch/text-classification/run_glue.py \
  --model_name_or_path models_128/order/start_${dataset2} \
  --train_file data/${dataset1}/train_${f}.json \
  --validation_file data/${dataset1}/eval.json \
  --output_dir models_128/order/start_${dataset2}_end_${dataset1}_${f} \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --do_train \
  --do_eval \
  --save_steps 100000000 \
  --evaluation_strategy steps \
  --eval_steps 300