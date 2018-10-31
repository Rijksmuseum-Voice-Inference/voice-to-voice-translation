import os

glr = [0.0001, 0.00001]
dlr = [0.0001, 0.00001]

command_1 = """CUDA_VISIBLE_DEVICES=0 python main.py --mode train --dataset allsstar --allsstar_index_file /data/felix/allsstar/data_toy_world/mgc_index.txt --image_size 256 --c_dim 22 --num_workers 8 --batch_size 8 --use_tensorboard TRUE --run_dir /data/felix/models/stargan --model_save_step 20000 --num_iters 1000000 \
--g_lr 0.0001 \
--d_lr 0.00001 \
--lambda_cls 10 \
--lambda_rec 10 \
--n_critic 1 \
--exp_name 0001glr_00001dlr_1t1critic_10lambdacls_10lambdarec_mean_nocompressG"""
