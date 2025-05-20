export CUDA_VISIBLE_DEVICES=0

model_name=TimeMixer

seq_len=96
e_layers=2
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=16
d_ff=32
train_epochs=10
patience=10
batch_size=128

for p in 96 192 336 720
do
  for l in TimeLagLoss
  do
    for a in 0 0.01 0.05 0.1 0.15 0.2
    do
      python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path  ./dataset/ETT-small/\
        --data_path ETTh1.csv \
        --model_id ETTh1_$seq_len'_'96 \
        --model $model_name \
        --data ETTh1 \
        --features M \
        --seq_len $seq_len \
        --label_len 0 \
        --pred_len $p \
        --e_layers $e_layers \
        --enc_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --itr 1 \
        --d_model $d_model \
        --d_ff $d_ff \
        --learning_rate $learning_rate \
        --train_epochs $train_epochs \
        --patience $patience \
        --batch_size 128 \
        --down_sampling_layers $down_sampling_layers \
        --down_sampling_method avg \
        --down_sampling_window $down_sampling_window \
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
        --root_path  ./dataset/ETT-small/\
        --data_path ETTh1.csv \
        --model_id ETTh1_$seq_len'_'96 \
        --model $model_name \
        --data ETTh1 \
        --features M \
        --seq_len $seq_len \
        --label_len 0 \
        --pred_len $p \
        --e_layers $e_layers \
        --enc_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --itr 1 \
        --d_model $d_model \
        --d_ff $d_ff \
        --learning_rate $learning_rate \
        --train_epochs $train_epochs \
        --patience $patience \
        --batch_size 128 \
        --down_sampling_layers $down_sampling_layers \
        --down_sampling_method avg \
        --down_sampling_window $down_sampling_window \
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
        --root_path  ./dataset/ETT-small/\
        --data_path ETTh1.csv \
        --model_id ETTh1_$seq_len'_'96 \
        --model $model_name \
        --data ETTh1 \
        --features M \
        --seq_len $seq_len \
        --label_len 0 \
        --pred_len $p \
        --e_layers $e_layers \
        --enc_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --itr 1 \
        --d_model $d_model \
        --d_ff $d_ff \
        --learning_rate $learning_rate \
        --train_epochs $train_epochs \
        --patience $patience \
        --batch_size 128 \
        --down_sampling_layers $down_sampling_layers \
        --down_sampling_method avg \
        --down_sampling_window $down_sampling_window \
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
        --root_path  ./dataset/ETT-small/\
        --data_path ETTh1.csv \
        --model_id ETTh1_$seq_len'_'96 \
        --model $model_name \
        --data ETTh1 \
        --features M \
        --seq_len $seq_len \
        --label_len 0 \
        --pred_len $p \
        --e_layers $e_layers \
        --enc_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --itr 1 \
        --d_model $d_model \
        --d_ff $d_ff \
        --learning_rate $learning_rate \
        --train_epochs $train_epochs \
        --patience $patience \
        --batch_size 128 \
        --down_sampling_layers $down_sampling_layers \
        --down_sampling_method avg \
        --down_sampling_window $down_sampling_window \
        --top_k 20 \
        --loss $l \
        --alpha $a
    done
  done
done

for p in 96 192 336 720
do
  for l in TimeLagLoss
  do
    for a in 0 0.01 0.05 0.1 0.15 0.2 0.25 0.3 0.4 0.5
    do
      python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path  ./dataset/ETT-small/\
        --data_path ETTh2.csv \
        --model_id ETTh2_$seq_len'_'96 \
        --model $model_name \
        --data ETTh2 \
        --features M \
        --seq_len $seq_len \
        --label_len 0 \
        --pred_len $p \
        --e_layers $e_layers \
        --enc_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --itr 1 \
        --d_model $d_model \
        --d_ff $d_ff \
        --learning_rate $learning_rate \
        --train_epochs $train_epochs \
        --patience $patience \
        --batch_size 128 \
        --down_sampling_layers $down_sampling_layers \
        --down_sampling_method avg \
        --down_sampling_window $down_sampling_window \
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
        --root_path  ./dataset/ETT-small/\
        --data_path ETTh2.csv \
        --model_id ETTh2_$seq_len'_'96 \
        --model $model_name \
        --data ETTh2 \
        --features M \
        --seq_len $seq_len \
        --label_len 0 \
        --pred_len $p \
        --e_layers $e_layers \
        --enc_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --itr 1 \
        --d_model $d_model \
        --d_ff $d_ff \
        --learning_rate $learning_rate \
        --train_epochs $train_epochs \
        --patience $patience \
        --batch_size 128 \
        --down_sampling_layers $down_sampling_layers \
        --down_sampling_method avg \
        --down_sampling_window $down_sampling_window \
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
        --root_path  ./dataset/ETT-small/\
        --data_path ETTh2.csv \
        --model_id ETTh2_$seq_len'_'96 \
        --model $model_name \
        --data ETTh2 \
        --features M \
        --seq_len $seq_len \
        --label_len 0 \
        --pred_len $p \
        --e_layers $e_layers \
        --enc_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --itr 1 \
        --d_model $d_model \
        --d_ff $d_ff \
        --learning_rate $learning_rate \
        --train_epochs $train_epochs \
        --patience $patience \
        --batch_size 128 \
        --down_sampling_layers $down_sampling_layers \
        --down_sampling_method avg \
        --down_sampling_window $down_sampling_window \
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
        --root_path  ./dataset/ETT-small/\
        --data_path ETTh2.csv \
        --model_id ETTh2_$seq_len'_'96 \
        --model $model_name \
        --data ETTh2 \
        --features M \
        --seq_len $seq_len \
        --label_len 0 \
        --pred_len $p \
        --e_layers $e_layers \
        --enc_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --itr 1 \
        --d_model $d_model \
        --d_ff $d_ff \
        --learning_rate $learning_rate \
        --train_epochs $train_epochs \
        --patience $patience \
        --batch_size 128 \
        --down_sampling_layers $down_sampling_layers \
        --down_sampling_method avg \
        --down_sampling_window $down_sampling_window \
        --top_k 20 \
        --loss $l \
        --alpha $a
    done
  done
done

for p in 96 192 336 720
do
  for l in TimeLagLoss
  do
    for a in 0 0.01 0.05 0.1 0.15 0.2
    do
      python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path  ./dataset/ETT-small/\
        --data_path ETTm1.csv \
        --model_id ETTm1_$seq_len'_'96 \
        --model $model_name \
        --data ETTm1 \
        --features M \
        --seq_len $seq_len \
        --label_len 0 \
        --pred_len $p \
        --e_layers $e_layers \
        --enc_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --itr 1 \
        --d_model $d_model \
        --d_ff $d_ff \
        --learning_rate $learning_rate \
        --train_epochs $train_epochs \
        --patience $patience \
        --batch_size 128 \
        --down_sampling_layers $down_sampling_layers \
        --down_sampling_method avg \
        --down_sampling_window $down_sampling_window \
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
        --root_path  ./dataset/ETT-small/\
        --data_path ETTm1.csv \
        --model_id ETTm1_$seq_len'_'96 \
        --model $model_name \
        --data ETTm1 \
        --features M \
        --seq_len $seq_len \
        --label_len 0 \
        --pred_len $p \
        --e_layers $e_layers \
        --enc_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --itr 1 \
        --d_model $d_model \
        --d_ff $d_ff \
        --learning_rate $learning_rate \
        --train_epochs $train_epochs \
        --patience $patience \
        --batch_size 128 \
        --down_sampling_layers $down_sampling_layers \
        --down_sampling_method avg \
        --down_sampling_window $down_sampling_window \
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
        --root_path  ./dataset/ETT-small/\
        --data_path ETTm1.csv \
        --model_id ETTm1_$seq_len'_'96 \
        --model $model_name \
        --data ETTm1 \
        --features M \
        --seq_len $seq_len \
        --label_len 0 \
        --pred_len $p \
        --e_layers $e_layers \
        --enc_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --itr 1 \
        --d_model $d_model \
        --d_ff $d_ff \
        --learning_rate $learning_rate \
        --train_epochs $train_epochs \
        --patience $patience \
        --batch_size 128 \
        --down_sampling_layers $down_sampling_layers \
        --down_sampling_method avg \
        --down_sampling_window $down_sampling_window \
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
        --root_path  ./dataset/ETT-small/\
        --data_path ETTm1.csv \
        --model_id ETTm1_$seq_len'_'96 \
        --model $model_name \
        --data ETTm1 \
        --features M \
        --seq_len $seq_len \
        --label_len 0 \
        --pred_len $p \
        --e_layers $e_layers \
        --enc_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --itr 1 \
        --d_model $d_model \
        --d_ff $d_ff \
        --learning_rate $learning_rate \
        --train_epochs $train_epochs \
        --patience $patience \
        --batch_size 128 \
        --down_sampling_layers $down_sampling_layers \
        --down_sampling_method avg \
        --down_sampling_window $down_sampling_window \
        --top_k 20 \
        --loss $l \
        --alpha $a
    done
  done
done

for p in 96 192 336 720
do
  for l in TimeLagLoss
  do
    for a in 0 0.01 0.05 0.1 0.15 0.2
    do
      python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path  ./dataset/ETT-small/\
        --data_path ETTm2.csv \
        --model_id ETTm2_$seq_len'_'96 \
        --model $model_name \
        --data ETTm2 \
        --features M \
        --seq_len $seq_len \
        --label_len 0 \
        --pred_len $p \
        --e_layers $e_layers \
        --enc_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --itr 1 \
        --d_model $d_model \
        --d_ff $d_ff \
        --learning_rate $learning_rate \
        --train_epochs $train_epochs \
        --patience $patience \
        --batch_size 128 \
        --down_sampling_layers $down_sampling_layers \
        --down_sampling_method avg \
        --down_sampling_window $down_sampling_window \
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
        --root_path  ./dataset/ETT-small/\
        --data_path ETTm2.csv \
        --model_id ETTm2_$seq_len'_'96 \
        --model $model_name \
        --data ETTm2 \
        --features M \
        --seq_len $seq_len \
        --label_len 0 \
        --pred_len $p \
        --e_layers $e_layers \
        --enc_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --itr 1 \
        --d_model $d_model \
        --d_ff $d_ff \
        --learning_rate $learning_rate \
        --train_epochs $train_epochs \
        --patience $patience \
        --batch_size 128 \
        --down_sampling_layers $down_sampling_layers \
        --down_sampling_method avg \
        --down_sampling_window $down_sampling_window \
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
        --root_path  ./dataset/ETT-small/\
        --data_path ETTm2.csv \
        --model_id ETTm2_$seq_len'_'96 \
        --model $model_name \
        --data ETTm2 \
        --features M \
        --seq_len $seq_len \
        --label_len 0 \
        --pred_len $p \
        --e_layers $e_layers \
        --enc_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --itr 1 \
        --d_model $d_model \
        --d_ff $d_ff \
        --learning_rate $learning_rate \
        --train_epochs $train_epochs \
        --patience $patience \
        --batch_size 128 \
        --down_sampling_layers $down_sampling_layers \
        --down_sampling_method avg \
        --down_sampling_window $down_sampling_window \
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
        --root_path  ./dataset/ETT-small/\
        --data_path ETTm2.csv \
        --model_id ETTm2_$seq_len'_'96 \
        --model $model_name \
        --data ETTm2 \
        --features M \
        --seq_len $seq_len \
        --label_len 0 \
        --pred_len $p \
        --e_layers $e_layers \
        --enc_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --itr 1 \
        --d_model $d_model \
        --d_ff $d_ff \
        --learning_rate $learning_rate \
        --train_epochs $train_epochs \
        --patience $patience \
        --batch_size 128 \
        --down_sampling_layers $down_sampling_layers \
        --down_sampling_method avg \
        --down_sampling_window $down_sampling_window \
        --top_k 20 \
        --loss $l \
        --alpha $a
    done
  done
done