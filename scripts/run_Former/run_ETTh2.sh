export CUDA_VISIBLE_DEVICES=1

#cd ..

for model in Autoformer FEDformer iTransformer DRFormer PatchTST
do

for preLen in 96 192 336 720
do

# ETTh2
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --task_id Former_ETTh2 \
  --model $model \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --padding_patch None \
  --pred_len $preLen \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 512 \
  --itr 3

done

done