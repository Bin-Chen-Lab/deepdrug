CUDA_VISIBLE_DEVICES=3 python3 ./denoise_gan_baseline.py --data_root ../data/ --cuda --batch_size 2048 > log_denoise_gan_baseline.txt 2>&1 &
