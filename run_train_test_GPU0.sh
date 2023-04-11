python train.py \
--dataset_type 'ETRI' \
--exp_id 1331 \
--gpu_num 0 \
--multi_gpu 1 \
--num_cores 8 \
--num_epochs 50 \
--batch_size 8 \
--learning_rate 0.0005 \
--best_k 15 \
--alpha 0.0 \
--beta 0.5 \
--gamma 0.0 \
--roi_grid_size 2 \
--roi_num_grid 40 \
--limit_range 50 \
--l2_loss_type 1 \
--apply_cyclic_schedule 1 \
--is_train_w_nuscenes 1


python test_all.py \
--dataset_type 'ETRI' \
--model_name 'autove' \
--exp_id 1331 \
--gpu_num 0 \
--best_k 15 \
--is_test_all 1

