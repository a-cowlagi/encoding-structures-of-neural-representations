python3 -u projection_suppression/train_suppressed_models.py --dataset cifar10 --model_name wr --epochs 50 --batch_size 128 --window_size 200 --compress_mode covariance --project_suppress project