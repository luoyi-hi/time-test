model_name=CycleNet

model_type='linear'
seq_len=96

for l in TimeLagLoss
do
for pred_len in 96 192 336 720
do
for a in 0.0 0.01 0.05 0.1 0.15 0.20 0.25 0.5
do
    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path ETTh1.csv \
      --model_id ETTh1'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data ETTh1 \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --cycle 24 \
      --model_type $model_type \
      --train_epochs 30 \
      --patience 5 \
      --itr 1 --batch_size 256 --learning_rate 0.01 --random_seed 2021 --loss $l --alpha $a
done
done
done

for l in TimeLagLoss
do
for pred_len in 96 192 336 720
do
for a in 0.0 0.01 0.05 0.1 0.15 0.20 0.25 0.5
do
    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path ETTh2.csv \
      --model_id ETTh2'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data ETTh2 \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --cycle 24 \
      --model_type $model_type \
      --train_epochs 30 \
      --patience 5 \
      --itr 1 --batch_size 256 --learning_rate 0.01 --random_seed 2021 --loss $l --alpha $a
done
done
done

for l in TimeLagLoss
do
for pred_len in 96 192 336 720
do
for a in 0.0 0.01 0.05 0.1 0.15 0.20 0.25 0.5
do
    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path ETTm1.csv \
      --model_id ETTm1'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data ETTm1 \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --cycle 96 \
      --model_type $model_type \
      --train_epochs 30 \
      --patience 5 \
      --itr 1 --batch_size 256 --learning_rate 0.01 --random_seed 2021 --loss $l --alpha $a
done
done
done

for l in TimeLagLoss
do
for pred_len in 96 192 336 720
do
for a in 0.0 0.01 0.05 0.1 0.15 0.20 0.25 0.5
do
    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path ETTm2.csv \
      --model_id ETTm2'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data ETTm2 \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --cycle 96 \
      --model_type $model_type \
      --train_epochs 30 \
      --patience 5 \
      --itr 1 --batch_size 256 --learning_rate 0.01 --random_seed 2021 --loss $l --alpha $a
done
done
done

for l in TimeLagLoss
do
for pred_len in 96 192 336 720
do
for a in 0.0 0.01 0.05 0.1 0.15 0.20 0.25 0.5
do
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path weather.csv \
    --model_id Weather'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 21 \
    --cycle 144 \
    --model_type $model_type \
    --train_epochs 30 \
    --patience 5 \
    --itr 1 --batch_size 256 --learning_rate 0.01 --random_seed 2021 --loss $l --alpha $a
done
done
done

for l in TimeLagLoss
do
for pred_len in 96 192 336 720
do
for a in 0.0 0.01 0.05 0.1 0.15 0.20 0.25 0.5
do
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path electricity.csv \
    --model_id electricity.csv'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 321 \
    --cycle 168 \
    --model_type $model_type \
    --train_epochs 30 \
    --patience 5 \
    --itr 1 --batch_size 64 --learning_rate 0.01 --random_seed 2021 --loss $l --alpha $a
done
done
done

for l in MSE MAE TILDE-Q TDTAlign
do
for pred_len in 96 192 336 720
do
for a in 1
do
    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path ETTh1.csv \
      --model_id ETTh1'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data ETTh1 \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --cycle 24 \
      --model_type $model_type \
      --train_epochs 30 \
      --patience 5 \
      --itr 1 --batch_size 256 --learning_rate 0.01 --random_seed 2021 --loss $l --alpha $a
done
done
done

for l in MSE MAE TILDE-Q TDTAlign
do
for pred_len in 96 192 336 720
do
for a in 1
do
    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path ETTh2.csv \
      --model_id ETTh2'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data ETTh2 \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --cycle 24 \
      --model_type $model_type \
      --train_epochs 30 \
      --patience 5 \
      --itr 1 --batch_size 256 --learning_rate 0.01 --random_seed 2021 --loss $l --alpha $a
done
done
done

for l in MSE MAE TILDE-Q TDTAlign
do
for pred_len in 96 192 336 720
do
for a in 1
do
    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path ETTm1.csv \
      --model_id ETTm1'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data ETTm1 \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --cycle 96 \
      --model_type $model_type \
      --train_epochs 30 \
      --patience 5 \
      --itr 1 --batch_size 256 --learning_rate 0.01 --random_seed 2021 --loss $l --alpha $a
done
done
done

for l in MSE MAE TILDE-Q TDTAlign
do
for pred_len in 96 192 336 720
do
for a in 1
do
    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path ETTm2.csv \
      --model_id ETTm2'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data ETTm2 \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --cycle 96 \
      --model_type $model_type \
      --train_epochs 30 \
      --patience 5 \
      --itr 1 --batch_size 256 --learning_rate 0.01 --random_seed 2021 --loss $l --alpha $a
done
done
done

for l in MSE MAE TILDE-Q TDTAlign
do
for pred_len in 96 192 336 720
do
for a in 1
do
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path weather.csv \
    --model_id Weather'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 21 \
    --cycle 144 \
    --model_type $model_type \
    --train_epochs 30 \
    --patience 5 \
    --itr 1 --batch_size 256 --learning_rate 0.01 --random_seed 2021 --loss $l --alpha $a
done
done
done

for l in MSE MAE TILDE-Q TDTAlign
do
for pred_len in 96 192 336 720
do
for a in 1
do
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path electricity.csv \
    --model_id electricity.csv'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 321 \
    --cycle 168 \
    --model_type $model_type \
    --train_epochs 30 \
    --patience 5 \
    --itr 1 --batch_size 64 --learning_rate 0.01 --random_seed 2021 --loss $l --alpha $a
done
done
done

for l in FreDF
do
for pred_len in 96 192 336 720
do
for a in 0.25 0.5 0.75 1
do
    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path ETTh1.csv \
      --model_id ETTh1'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data ETTh1 \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --cycle 24 \
      --model_type $model_type \
      --train_epochs 30 \
      --patience 5 \
      --itr 1 --batch_size 256 --learning_rate 0.01 --random_seed 2021 --loss $l --alpha $a
done
done
done

for l in FreDF
do
for pred_len in 96 192 336 720
do
for a in 0.25 0.5 0.75 1
do
    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path ETTh2.csv \
      --model_id ETTh2'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data ETTh2 \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --cycle 24 \
      --model_type $model_type \
      --train_epochs 30 \
      --patience 5 \
      --itr 1 --batch_size 256 --learning_rate 0.01 --random_seed 2021 --loss $l --alpha $a
done
done
done

for l in FreDF
do
for pred_len in 96 192 336 720
do
for a in 0.25 0.5 0.75 1
do
    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path ETTm1.csv \
      --model_id ETTm1'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data ETTm1 \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --cycle 96 \
      --model_type $model_type \
      --train_epochs 30 \
      --patience 5 \
      --itr 1 --batch_size 256 --learning_rate 0.01 --random_seed 2021 --loss $l --alpha $a
done
done
done

for l in FreDF
do
for pred_len in 96 192 336 720
do
for a in 0.25 0.5 0.75 1
do
    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path ETTm2.csv \
      --model_id ETTm2'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data ETTm2 \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --cycle 96 \
      --model_type $model_type \
      --train_epochs 30 \
      --patience 5 \
      --itr 1 --batch_size 256 --learning_rate 0.01 --random_seed 2021 --loss $l --alpha $a
done
done
done

for l in FreDF
do
for pred_len in 96 192 336 720
do
for a in 0.25 0.5 0.75 1
do
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path weather.csv \
    --model_id Weather'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 21 \
    --cycle 144 \
    --model_type $model_type \
    --train_epochs 30 \
    --patience 5 \
    --itr 1 --batch_size 256 --learning_rate 0.01 --random_seed 2021 --loss $l --alpha $a
done
done
done

for l in FreDF
do
for pred_len in 96 192 336 720
do
for a in 0.25 0.5 0.75 1
do
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path electricity.csv \
    --model_id electricity.csv'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 321 \
    --cycle 168 \
    --model_type $model_type \
    --train_epochs 30 \
    --patience 5 \
    --itr 1 --batch_size 64 --learning_rate 0.01 --random_seed 2021 --loss $l --alpha $a
done
done
done

for l in PS_Loss
do
for pred_len in 96 192 336 720
do
for a in 1 3 5 10
do
    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path ETTh1.csv \
      --model_id ETTh1'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data ETTh1 \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --cycle 24 \
      --model_type $model_type \
      --train_epochs 30 \
      --patience 5 \
      --itr 1 --batch_size 256 --learning_rate 0.01 --random_seed 2021 --loss $l --alpha $a
done
done
done

for l in PS_Loss
do
for pred_len in 96 192 336 720
do
for a in 1 3 5 10
do
    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path ETTh2.csv \
      --model_id ETTh2'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data ETTh2 \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --cycle 24 \
      --model_type $model_type \
      --train_epochs 30 \
      --patience 5 \
      --itr 1 --batch_size 256 --learning_rate 0.01 --random_seed 2021 --loss $l --alpha $a
done
done
done

for l in PS_Loss
do
for pred_len in 96 192 336 720
do
for a in 1 3 5 10
do
    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path ETTm1.csv \
      --model_id ETTm1'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data ETTm1 \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --cycle 96 \
      --model_type $model_type \
      --train_epochs 30 \
      --patience 5 \
      --itr 1 --batch_size 256 --learning_rate 0.01 --random_seed 2021 --loss $l --alpha $a
done
done
done

for l in PS_Loss
do
for pred_len in 96 192 336 720
do
for a in 1 3 5 10
do
    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path ETTm2.csv \
      --model_id ETTm2'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data ETTm2 \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --cycle 96 \
      --model_type $model_type \
      --train_epochs 30 \
      --patience 5 \
      --itr 1 --batch_size 256 --learning_rate 0.01 --random_seed 2021 --loss $l --alpha $a
done
done
done

for l in PS_Loss
do
for pred_len in 96 192 336 720
do
for a in 1 3 5 10
do
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path weather.csv \
    --model_id Weather'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 21 \
    --cycle 144 \
    --model_type $model_type \
    --train_epochs 30 \
    --patience 5 \
    --itr 1 --batch_size 256 --learning_rate 0.01 --random_seed 2021 --loss $l --alpha $a
done
done
done

for l in PS_Loss
do
for pred_len in 96 192 336 720
do
for a in 1 3 5 10
do
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path electricity.csv \
    --model_id electricity.csv'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 321 \
    --cycle 168 \
    --model_type $model_type \
    --train_epochs 30 \
    --patience 5 \
    --itr 1 --batch_size 64 --learning_rate 0.01 --random_seed 2021 --loss $l --alpha $a
done
done
done