export CUDA_VISIBLE_DEVICES=0

#cd ..

for model in Ablation
do

for preLen in 24 36 48 60
do
# illness
python -u run.py \
 --is_training 1 \
 --root_path ./dataset/illness/ \
 --data_path national_illness.csv \
 --task_id Test1_ill \
 --model $model \
 --data custom \
 --features M \
 --seq_len 36 \
 --label_len 18 \
 --padding_patch None \
 --freq_topk 10 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 7 \
 --dec_in 7 \
 --c_out 7 \
 --des 'Exp' \
 --itr 3
done


for preLen in 96 192 336 720
do

# ETTh1
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --task_id Test1_ETTh1 \
  --model $model \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $preLen \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --use_fan False \
  --des 'Exp' \
  --d_model 512 \
  --itr 3
done


for preLen in 96 192 336 720
do

# weather
python -u run.py \
 --is_training 1 \
 --root_path ./dataset/weather/ \
 --data_path weather.csv \
 --task_id Test1_weather \
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
 --enc_in 21 \
 --dec_in 21 \
 --c_out 21 \
 --des 'Exp' \
 --itr 3
done

for preLen in 96 192 336 720
do
python -u run.py \
 --is_training 1 \
 --root_path ./dataset/electricity/ \
 --data_path electricity.csv \
 --task_id Test1_ECL \
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

