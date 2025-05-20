export CUDA_VISIBLE_DEVICES=0

model_name=SOFTS

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
        --seq_len 96 \
        --label_len 48 \
        --pred_len $p \
        --e_layers 3 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 21 \
        --dec_in 21 \
        --c_out 21 \
        --des 'Exp' \
        --d_model 512\
        --d_ff 512\
        --itr 1 \
        --learning_rate 0.0005 \
        --batch_size 128 \
        --top_k 20 \
        --loss $l \
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
        --seq_len 96 \
        --label_len 48 \
        --pred_len $p \
        --e_layers 3 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 21 \
        --dec_in 21 \
        --c_out 21 \
        --des 'Exp' \
        --d_model 512\
        --d_ff 512\
        --itr 1 \
        --learning_rate 0.0005 \
        --batch_size 128 \
        --top_k 20 \
        --loss $l \
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
        --seq_len 96 \
        --label_len 48 \
        --pred_len $p \
        --e_layers 3 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 21 \
        --dec_in 21 \
        --c_out 21 \
        --des 'Exp' \
        --d_model 512\
        --d_ff 512\
        --itr 1 \
        --learning_rate 0.0005 \
        --batch_size 128 \
        --top_k 20 \
        --loss $l \
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
        --seq_len 96 \
        --label_len 48 \
        --pred_len $p \
        --e_layers 3 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 21 \
        --dec_in 21 \
        --c_out 21 \
        --des 'Exp' \
        --d_model 512\
        --d_ff 512\
        --itr 1 \
        --learning_rate 0.0005 \
        --batch_size 128 \
        --top_k 20 \
        --loss $l \
        --alpha $a
    done
  done
done