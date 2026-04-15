export CUDA_VISIBLE_DEVICES=2

#cd ..

for model in MICN
do

for preLen in 96 192 336 720
do

# electricity
python -u run.py \
 --is_training 1 \
 --root_path ./dataset/electricity/ \
 --data_path electricity.csv \
 --task_id CNN_ECL \
 --model $model \
 --data custom \
 --features M \
 --seq_len 96 \
 --label_len 48 \
 --padding_patch None \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 321 \
 --dec_in 321 \
 --c_out 321 \
 --des 'Exp' \
 --itr 3

 done

 done