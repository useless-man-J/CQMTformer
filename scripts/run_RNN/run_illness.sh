export CUDA_VISIBLE_DEVICES=1

#cd ..

for model in S_Mamba BiMamba4TS
do

for preLen in 24 36 48 60
do
# illness
python -u run.py \
 --is_training 1 \
 --root_path ./dataset/illness/ \
 --data_path national_illness.csv \
 --task_id RNN_ili \
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

