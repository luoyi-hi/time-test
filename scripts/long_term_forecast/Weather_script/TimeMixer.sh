export CUDA_VISIBLE_DEVICES=0

model_name=TimeMixer

seq_len=96
e_layers=3
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=16
d_ff=32
train_epochs=20
patience=10

for p in 96 192 336 720
do
  for l in TimeLagLoss
  do
    for a in 0 0.01 0.05 0.1 0.15 0.2
    do
      python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./dataset/weather/ \
        --data_path weather.csv \
        --model_id weather_96_96 \
        --model $model_name \
        --data custom \
        --features M \
        --seq_len $seq_len \
        --label_len 0 \
        --pred_len $p \
        --e_layers $e_layers \
        --d_layers 1 \
        --factor 3 \
        --enc_in 21 \
        --dec_in 21 \
        --c_out 21 \
        --des 'Exp' \
        --itr 1 \
        --d_model $d_model \
        --d_ff $d_ff \
        --batch_size 128 \
        --top_k 20 \
        --loss $l \
        --learning_rate $learning_rate \
        --train_epochs $train_epochs \
        --patience $patience \
        --down_sampling_layers $down_sampling_layers \
        --down_sampling_method avg \
        --down_sampling_window $down_sampling_window \
        --alpha $a
    done
  done
done

for p in 96 192 336 720
do
  for l in MSE MAE TILDE-Q TDTAlign
  do
    for a in 1
    do
      python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./dataset/weather/ \
        --data_path weather.csv \
        --model_id weather_96_96 \
        --model $model_name \
        --data custom \
        --features M \
        --seq_len $seq_len \
        --label_len 0 \
        --pred_len $p \
        --e_layers $e_layers \
        --d_layers 1 \
        --factor 3 \
        --enc_in 21 \
        --dec_in 21 \
        --c_out 21 \
        --des 'Exp' \
        --itr 1 \
        --d_model $d_model \
        --d_ff $d_ff \
        --batch_size 128 \
        --top_k 20 \
        --loss $l \
        --learning_rate $learning_rate \
        --train_epochs $train_epochs \
        --patience $patience \
        --down_sampling_layers $down_sampling_layers \
        --down_sampling_method avg \
        --down_sampling_window $down_sampling_window \
        --alpha $a
    done
  done
done

for p in 96 192 336 720
do
  for l in FreDF
  do
    for a in 0.25 0.5 0.75 1
    do
      python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./dataset/weather/ \
        --data_path weather.csv \
        --model_id weather_96_96 \
        --model $model_name \
        --data custom \
        --features M \
        --seq_len $seq_len \
        --label_len 0 \
        --pred_len $p \
        --e_layers $e_layers \
        --d_layers 1 \
        --factor 3 \
        --enc_in 21 \
        --dec_in 21 \
        --c_out 21 \
        --des 'Exp' \
        --itr 1 \
        --d_model $d_model \
        --d_ff $d_ff \
        --batch_size 128 \
        --top_k 20 \
        --loss $l \
        --learning_rate $learning_rate \
        --train_epochs $train_epochs \
        --patience $patience \
        --down_sampling_layers $down_sampling_layers \
        --down_sampling_method avg \
        --down_sampling_window $down_sampling_window \
        --alpha $a
    done
  done
done

for p in 96 192 336 720
do
  for l in PS_Loss
  do
    for a in 1 3 5 10
    do
      python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./dataset/weather/ \
        --data_path weather.csv \
        --model_id weather_96_96 \
        --model $model_name \
        --data custom \
        --features M \
        --seq_len $seq_len \
        --label_len 0 \
        --pred_len $p \
        --e_layers $e_layers \
        --d_layers 1 \
        --factor 3 \
        --enc_in 21 \
        --dec_in 21 \
        --c_out 21 \
        --des 'Exp' \
        --itr 1 \
        --d_model $d_model \
        --d_ff $d_ff \
        --batch_size 128 \
        --top_k 20 \
        --loss $l \
        --learning_rate $learning_rate \
        --train_epochs $train_epochs \
        --patience $patience \
        --down_sampling_layers $down_sampling_layers \
        --down_sampling_method avg \
        --down_sampling_window $down_sampling_window \
        --alpha $a
    done
  done
done
