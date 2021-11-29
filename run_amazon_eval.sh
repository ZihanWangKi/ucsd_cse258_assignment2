gpu=$1
dataset=$2
odataset=$3

for f in "100" "200" "500"; do
  CUDA_VISIBLE_DEVICES=${gpu} python transformers/examples/pytorch/text-classification/run_glue.py \
    --model_name_or_path bert-base-cased \
    --train_file data/${dataset}/train_${f}.json \
    --validation_file data/${dataset}/eval.json \
    --output_dir models_128/${dataset}_${f} \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --do_train \
    --do_eval \
    --save_steps 100000000 \
    --evaluation_strategy steps \
    --eval_steps 300
done

for f in "100" "200" "500"; do
  CUDA_VISIBLE_DEVICES=${gpu} python transformers/examples/pytorch/text-classification/run_glue.py \
    --model_name_or_path bert-base-cased \
    --train_file data/concat_${dataset}_${f}_${odataset}/train.json \
    --validation_file data/${dataset}/eval.json \
    --output_dir models_128/${dataset}_${f}_${odataset} \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --do_train \
    --do_eval \
    --save_steps 100000000 \
    --evaluation_strategy steps \
    --eval_steps 300
done