export CUDA_VISIBLE_DEVICES=0

for model in CQMTformer
do

for preLen in 96 192 336 720
do

# electricity
python -u run.py \
 --is_training 1 \
 --root_path ./dataset/electricity/ \
 --data_path electricity.csv \
 --task_id MyModel_ECL \
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

for preLen in 96 192 336 720
do

# ETTh1
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --task_id MyModel_ETTh1 \
  --model $model \
  --data ETTh1 \
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

for preLen in 96 192 336 720
do

# ETTh2
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --task_id MyModel_ETTh2 \
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

for preLen in 96 192 336 720
do

# ETTm1
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --task_id MyModel_ETTm1 \
  --model $model \
  --data ETTm1 \
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
  --itr 3 \

done

for preLen in 96 192 336 720
do

# ETTm2
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --task_id MyModel_ETTm2 \
  --model $model \
  --data ETTm2 \
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

for preLen in 96 192 336 720
do

# exchange
python -u run.py \
 --is_training 1 \
 --root_path ./dataset/exchange_rate/ \
 --data_path exchange_rate.csv \
 --task_id MyModel_Exchange \
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
 --enc_in 8 \
 --dec_in 8 \
 --c_out 8 \
 --des 'Exp' \
 --itr 3
done

for preLen in 96 192 336 720
do

# traffic
python -u run.py \
 --is_training 1 \
 --root_path ./dataset/traffic/ \
 --data_path traffic.csv \
 --task_id MyModel_traffic \
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
 --enc_in 862 \
 --dec_in 862 \
 --c_out 862 \
 --des 'Exp' \
 --itr 3 \
 --train_epochs 5

done

for preLen in 96 192 336 720
do

# weather
python -u run.py \
 --is_training 1 \
 --root_path ./dataset/weather/ \
 --data_path weather.csv \
 --task_id MyModel_weather \
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

for preLen in 24 36 48 60
do
# illness
python -u run.py \
 --is_training 1 \
 --root_path ./dataset/illness/ \
 --data_path national_illness.csv \
 --task_id MyModel_ili \
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

done