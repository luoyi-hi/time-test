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
        --root_path ./dataset/ETT-small/ \
        --data_path ETTh1.csv \
        --model_id ETTh1_96_96 \
        --model $model_name \
        --data ETTh1 \
        --features M \
        --batch_size 128 \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $p \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --d_model 512 \
        --d_ff 512 \
        --itr 1 \
        --learning_rate 0.0001 \
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
        --root_path ./dataset/ETT-small/ \
        --data_path ETTh1.csv \
        --model_id ETTh1_96_96 \
        --model $model_name \
        --data ETTh1 \
        --features M \
        --batch_size 128 \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $p \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --d_model 512 \
        --d_ff 512 \
        --itr 1 \
        --learning_rate 0.0001 \
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
        --root_path ./dataset/ETT-small/ \
        --data_path ETTh1.csv \
        --model_id ETTh1_96_96 \
        --model $model_name \
        --data ETTh1 \
        --features M \
        --batch_size 128 \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $p \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --d_model 512 \
        --d_ff 512 \
        --itr 1 \
        --learning_rate 0.0001 \
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
        --root_path ./dataset/ETT-small/ \
        --data_path ETTh1.csv \
        --model_id ETTh1_96_96 \
        --model $model_name \
        --data ETTh1 \
        --features M \
        --batch_size 128 \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $p \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --d_model 512 \
        --d_ff 512 \
        --itr 1 \
        --learning_rate 0.0001 \
        --top_k 20 \
        --loss $l \
        --alpha $a
    done
  done
done

for p in 96 192
do
  for l in TimeLagLoss
  do
    for a in 0 0.01 0.05 0.1 0.15 0.2 0.25 0.3 0.4 0.5
    do
      python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./dataset/ETT-small/ \
        --data_path ETTh2.csv \
        --model_id ETTh2_96_96 \
        --model $model_name \
        --data ETTh2 \
        --features M \
        --batch_size 128 \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $p \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --d_model 512 \
        --d_ff 512 \
        --itr 1 \
        --learning_rate 0.0001 \
        --top_k 20 \
        --loss $l \
        --alpha $a
    done
  done
done

for p in 336 720
do
  for l in TimeLagLoss
  do
    for a in 0 0.01 0.05 0.1 0.15 0.2 0.25 0.3 0.4 0.5
    do
      python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./dataset/ETT-small/ \
        --data_path ETTh2.csv \
        --model_id ETTh2_96_96 \
        --model $model_name \
        --data ETTh2 \
        --features M \
        --batch_size 128 \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $p \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --d_model 512 \
        --d_ff 512 \
        --itr 1 \
        --learning_rate 0.0005 \
        --top_k 20 \
        --loss $l \
        --alpha $a
    done
  done
done

for p in 96 192
do
  for l in MSE MAE TILDE-Q TDTAlign
  do
    for a in 1
    do
      python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./dataset/ETT-small/ \
        --data_path ETTh2.csv \
        --model_id ETTh2_96_96 \
        --model $model_name \
        --data ETTh2 \
        --features M \
        --batch_size 128 \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $p \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --d_model 512 \
        --d_ff 512 \
        --itr 1 \
        --learning_rate 0.0001 \
        --top_k 20 \
        --loss $l \
        --alpha $a
    done
  done
done

for p in 336 720
do
  for l in MSE MAE TILDE-Q TDTAlign
  do
    for a in 1
    do
      python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./dataset/ETT-small/ \
        --data_path ETTh2.csv \
        --model_id ETTh2_96_96 \
        --model $model_name \
        --data ETTh2 \
        --features M \
        --batch_size 128 \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $p \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --d_model 512 \
        --d_ff 512 \
        --itr 1 \
        --learning_rate 0.0005 \
        --top_k 20 \
        --loss $l \
        --alpha $a
    done
  done
done

for p in 96 192
do
  for l in FreDF
  do
    for a in 0.25 0.5 0.75 1
    do
      python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./dataset/ETT-small/ \
        --data_path ETTh2.csv \
        --model_id ETTh2_96_96 \
        --model $model_name \
        --data ETTh2 \
        --features M \
        --batch_size 128 \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $p \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --d_model 512 \
        --d_ff 512 \
        --itr 1 \
        --learning_rate 0.0001 \
        --top_k 20 \
        --loss $l \
        --alpha $a
    done
  done
done

for p in 336 720
do
  for l in FreDF
  do
    for a in 0.25 0.5 0.75 1
    do
      python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./dataset/ETT-small/ \
        --data_path ETTh2.csv \
        --model_id ETTh2_96_96 \
        --model $model_name \
        --data ETTh2 \
        --features M \
        --batch_size 128 \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $p \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --d_model 512 \
        --d_ff 512 \
        --itr 1 \
        --learning_rate 0.0005 \
        --top_k 20 \
        --loss $l \
        --alpha $a
    done
  done
done

for p in 96 192
do
  for l in PS_Loss
  do
    for a in 1 3 5 10
    do
      python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./dataset/ETT-small/ \
        --data_path ETTh2.csv \
        --model_id ETTh2_96_96 \
        --model $model_name \
        --data ETTh2 \
        --features M \
        --batch_size 128 \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $p \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --d_model 512 \
        --d_ff 512 \
        --itr 1 \
        --learning_rate 0.0001 \
        --top_k 20 \
        --loss $l \
        --alpha $a
    done
  done
done

for p in 336 720
do
  for l in PS_Loss
  do
    for a in 1 3 5 10
    do
      python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./dataset/ETT-small/ \
        --data_path ETTh2.csv \
        --model_id ETTh2_96_96 \
        --model $model_name \
        --data ETTh2 \
        --features M \
        --batch_size 128 \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $p \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --d_model 512 \
        --d_ff 512 \
        --itr 1 \
        --learning_rate 0.0005 \
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
        --root_path ./dataset/ETT-small/ \
        --data_path ETTm1.csv \
        --model_id ETTm1_96_96 \
        --model $model_name \
        --data ETTm1 \
        --features M \
        --batch_size 128 \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $p \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --d_model 512 \
        --d_ff 512 \
        --itr 1 \
        --learning_rate 0.0005 \
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
        --root_path ./dataset/ETT-small/ \
        --data_path ETTm1.csv \
        --model_id ETTm1_96_96 \
        --model $model_name \
        --data ETTm1 \
        --features M \
        --batch_size 128 \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $p \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --d_model 512 \
        --d_ff 512 \
        --itr 1 \
        --learning_rate 0.0005 \
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
        --root_path ./dataset/ETT-small/ \
        --data_path ETTm1.csv \
        --model_id ETTm1_96_96 \
        --model $model_name \
        --data ETTm1 \
        --features M \
        --batch_size 128 \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $p \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --d_model 512 \
        --d_ff 512 \
        --itr 1 \
        --learning_rate 0.0005 \
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
        --root_path ./dataset/ETT-small/ \
        --data_path ETTm1.csv \
        --model_id ETTm1_96_96 \
        --model $model_name \
        --data ETTm1 \
        --features M \
        --batch_size 128 \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $p \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --d_model 512 \
        --d_ff 512 \
        --itr 1 \
        --learning_rate 0.0005 \
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
        --root_path ./dataset/ETT-small/ \
        --data_path ETTm2.csv \
        --model_id ETTm2_96_96 \
        --model $model_name \
        --data ETTm2 \
        --features M \
        --batch_size 128 \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $p \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --d_model 512 \
        --d_ff 512 \
        --itr 1 \
        --learning_rate 0.0005 \
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
        --root_path ./dataset/ETT-small/ \
        --data_path ETTm2.csv \
        --model_id ETTm2_96_96 \
        --model $model_name \
        --data ETTm2 \
        --features M \
        --batch_size 128 \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $p \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --d_model 512 \
        --d_ff 512 \
        --itr 1 \
        --learning_rate 0.0005 \
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
        --root_path ./dataset/ETT-small/ \
        --data_path ETTm2.csv \
        --model_id ETTm2_96_96 \
        --model $model_name \
        --data ETTm2 \
        --features M \
        --batch_size 128 \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $p \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --d_model 512 \
        --d_ff 512 \
        --itr 1 \
        --learning_rate 0.0005 \
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
        --root_path ./dataset/ETT-small/ \
        --data_path ETTm2.csv \
        --model_id ETTm2_96_96 \
        --model $model_name \
        --data ETTm2 \
        --features M \
        --batch_size 128 \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $p \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --d_model 512 \
        --d_ff 512 \
        --itr 1 \
        --learning_rate 0.0005 \
        --top_k 20 \
        --loss $l \
        --alpha $a
    done
  done
done
