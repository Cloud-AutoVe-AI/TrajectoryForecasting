import argparse

parser = argparse.ArgumentParser()

# Exp Info
parser.add_argument('--model_name', type=str, default='')
# -----------------------------------
# 'vehicle' or 'pedestrian'
# -----------------------------------
parser.add_argument('--model_mode', type=str, default='vehicle')
parser.add_argument('--exp_id', type=int, default=300)
parser.add_argument('--gpu_num', type=int, default=0)
parser.add_argument('--load_pretrained', type=int, default=0)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--multi_gpu', type=int, default=1)
parser.add_argument('--num_cores', type=int, default=1)


# Dataset
parser.add_argument('--dataset_path', type=str, default='')
# -----------------------------------
# 'argoverse' or 'nuscenes' or 'ETRI'
# -----------------------------------
parser.add_argument('--dataset_type', type=str, default='ETRI')
parser.add_argument('--preprocess_trajectory', type=int, default=0)
parser.add_argument('--num_turn_scene_repeats', type=int, default=0)
parser.add_argument('--input_dim', type=int, default=2)
parser.add_argument('--scene_accept_prob', type=float, default=1.0)
parser.add_argument('--past_horizon_seconds', type=float, default=2)
# --------------------------------------------------------------
# future horizon |  Autove  |  Nuscenes | Argoverse |   NMP
# --------------------------------------------------------------
# vehicle        |    4     |     6     |     3     |   4
# pedestrian     |    4     |     -     |     -     |   -
# --------------------------------------------------------------
parser.add_argument('--future_horizon_seconds', type=float, default=4)
parser.add_argument('--min_past_horizon_seconds', type=float, default=1.5)
parser.add_argument('--min_future_horizon_seconds', type=float, default=3)
# ---------------------------------------------------
# sample period  |  Autove  |  Nuscenes | Argoverse |
# ---------------------------------------------------
# vehicle        |    2     |     2     |     5     |
# pedestrian     |    2     |     -     |     -     |
# ---------------------------------------------------
parser.add_argument('--target_sample_period', type=float, default=2)  # Hz ---
parser.add_argument('--val_ratio', type=float, default=0.05)
parser.add_argument('--max_num_agents', type=int, default=100)
parser.add_argument('--min_num_agents', type=int, default=2)
parser.add_argument('--stop_agents_remove_prob', type=float, default=0)
parser.add_argument('--limit_range_change_prob', type=float, default=0.0)
# -----------------------------------
# 0 : vehicle (Trajectron++)
# 1 : vehicle (Nuscenes benchmark)
# 2 : vehicle and pedestrian (Trajectron++)
# -----------------------------------
parser.add_argument('--category_filtering_method', type=int, default=2)

# Training Env
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--best_k', type=int, default=5)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--min_learning_rate', type=float, default=0.001)
parser.add_argument('--learning_rate_cnn', type=float, default=0.005)
parser.add_argument('--grad_clip', type=float, default=0.0)
parser.add_argument('--n_cycle', type=int, default=4)
parser.add_argument('--warmup_ratio', type=float, default=0.5)
parser.add_argument('--h_noise_std', type=float, default=4.0)

parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--beta', type=float, default=1.0)
parser.add_argument('--gamma', type=float, default=1.0)
parser.add_argument('--kappa', type=float, default=1.0)

parser.add_argument('--valid_step', type=int, default=1)
parser.add_argument('--save_every', type=int, default=3)
parser.add_argument('--max_num_chkpts', type=int, default=5)

parser.add_argument('--apply_cyclic_schedule', type=int, default=0)
parser.add_argument('--separate_lr_for_cnn', type=int, default=0)
parser.add_argument('--apply_lr_scheduling', type=int, default=0)
parser.add_argument('--use_preprocessed_lidar', type=int, default=0)
# -------------------------------------------------------------------------
#   unit: m    | ilvm | covernet | autove | autove_ped | scratch |  NMP
# -------------------------------------------------------------------------
# limit range  |  50  |   150    |   50   |     90     |   50    |  70
# xy_min_max   |  70  |    ?     |   90   |     90     |   100   |  70
# -------------------------------------------------------------------------
parser.add_argument('--limit_range', type=int, default=50)
parser.add_argument('--x_range_min', type=float, default=-90)
parser.add_argument('--x_range_max', type=float, default=90)
parser.add_argument('--y_range_min', type=float, default=-90)
parser.add_argument('--y_range_max', type=float, default=90)
parser.add_argument('--z_range_min', type=float, default=-1)
parser.add_argument('--z_range_max', type=float, default=4)
parser.add_argument('--num_lidar_sweeps', type=int, default=5)
# ------------------------------------------------------------------------------------
#                 |  ilvm   | covernet | autove     | autove_ped | NMP     |
# ------------------------------------------------------------------------------------
# map size        |  700    |          |  600/900   | 600/900    | 600/700 |
# range           |  ?      |          |  120/180   | 120/180    | 120/140 |
# resol (m/pel)   |  0.2    |          |  0.2       |    0.2     | 0.2     |
# hdmap_ch_dim    |6(a)/7(n)|   3(n)   |  4(n,e)    |   4(n,e)   | 4(n,e)  |
# feat_map_size   |   ?     |   4096   |  150/225   |   150/225  | 150/175 |
# ------------------------------------------------------------------------------------
parser.add_argument('--map_size', type=int, default=900)
parser.add_argument('--hdmap_type', type=int, default=0) # 0:color, 1:multi-ch
parser.add_argument('--hdmap_ch_dim', type=int, default=4)
parser.add_argument('--feat_map_size', type=int, default=225)
parser.add_argument('--lidar_map_ch_dim', type=int, default=25)
parser.add_argument('--centerline_width', type=int, default=2)

def ilvm(parser):

    parser.add_argument('--heading_est_method', type=int, default=1)  # 0: traj, 1: bbox
    parser.add_argument('--add_noise_for_heading_est', type=int, default=0)

    parser.add_argument('--is_draw_centerlines', type=int, default=1)
    parser.add_argument('--is_gauss_prior', type=int, default=0)
    parser.add_argument('--use_past_traj', type=int, default=1)

    # Agent Feature Extraction
    parser.add_argument('--roi_grid_size', type=float, default=2)  # meter / pixel
    parser.add_argument('--roi_num_grid', type=float, default=40)  # 40 pixels

    # Traj Encoder
    parser.add_argument('--traj_enc_h_dim', type=int, default=64)

    # SIM Encoder
    parser.add_argument('--sim_enc_h_dim', type=int, default=64)

    # SIM Prior
    parser.add_argument('--sim_prior_h_dim', type=int, default=64)

    # Traj Decoder
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--trajdec_h_dim', type=int, default=256)

    args = parser.parse_args()

    return args

def covernet(parser):

    parser.add_argument('--use_original_gt_traj', type=int, default=0)

    args = parser.parse_args()

    return args

def autove(parser):


    parser.add_argument('--is_train_w_nuscenes', type=int, default=0)
    parser.add_argument('--is_train_dis', type=int, default=0)
    parser.add_argument('--is_use_proc_traj', type=int, default=0)
    parser.add_argument('--random_flip_prob', type=float, default=0.0)
    parser.add_argument('--add_heading_noise', type=int, default=0)
    parser.add_argument('--heading_noise_deg_std', type=float, default=0.0)
    parser.add_argument('--is_bbox_heading', type=int, default=1)
    parser.add_argument('--is_shift_lane_deg', type=int, default=0)
    parser.add_argument('--agent_sampling_prob', type=float, default=0.0)
    parser.add_argument('--agent_sampling_thr', type=float, default=0.0)
    parser.add_argument('--pos_noise_prob', type=float, default=0.0)
    parser.add_argument('--pos_noise_std', type=float, default=0.0) # update, 220511
    parser.add_argument('--lane_color_prob', type=float, default=0.0)
    parser.add_argument('--broken_line_prob', type=float, default=0)
    parser.add_argument('--broken_line_ratio', type=float, default=0)


    # ----------------------
    # average | WTA | EWTA |
    # ----------------------
    #    0    |  1  |  2   |
    # ----------------------

    parser.add_argument('--l2_loss_type', type=int, default=1)
    parser.add_argument('--EWTA_ratio', type=float, default=0.5)

    # Others
    parser.add_argument('--heading_speed_thr', type=float, default=3)

    # CNN
    parser.add_argument('--cnn_outch_dim', type=float, default=128)

    # Agent Feature Extraction
    parser.add_argument('--roi_grid_size', type=float, default=2)  # meter / pixel
    parser.add_argument('--roi_num_grid', type=int, default=40)  # pixels
    parser.add_argument('--agent_mapfeat_dim', type=float, default=256)

    # CVAE Encoder
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--cvae_enc_dim', type=int, default=64)

    # Traj Encoder, update, 220511
    parser.add_argument('--traj_enc_type', type=int, default=1, help='0: LSTM only, 1: Residual+LSTM')
    parser.add_argument('--traj_enc_h_dim', type=int, default=16)

    # Traj Decoder
    parser.add_argument('--traj_dec_h_dim', type=int, default=256)
    parser.add_argument('--dec_drop_prob', type=float, default=0.2)

    args = parser.parse_args()

    return args

def autove_ped(parser):

    parser.add_argument('--is_train_w_nuscenes', type=int, default=1)
    parser.add_argument('--is_train_dis', type=int, default=0)
    parser.add_argument('--is_data_conv', type=int, default=0)
    parser.add_argument('--is_random_rotate', type=int, default=0)

    # -----------------------
    # average | WTA | EWTA |
    # ----------------------
    #    0    |  1  |  2   |
    # -----------------------

    parser.add_argument('--l2_loss_type', type=int, default=1)
    parser.add_argument('--EWTA_ratio', type=float, default=0.5)

    # CNN
    parser.add_argument('--cnn_outch_dim', type=float, default=128)

    # Agent Feature Extraction
    parser.add_argument('--roi_grid_size', type=float, default=2.0)  # meter / pixel, 1
    parser.add_argument('--roi_num_grid', type=int, default=40)  # pixels, 40
    parser.add_argument('--agent_mapfeat_dim', type=float, default=256)

    # CVAE Encoder
    parser.add_argument('--latent_dim', type=int, default=16)
    parser.add_argument('--cvae_enc_dim', type=int, default=16)

    # Traj Encoder
    parser.add_argument('--traj_enc_h_dim', type=int, default=16)

    # Traj Decoder
    parser.add_argument('--traj_dec_h_dim', type=int, default=256)
    parser.add_argument('--dec_drop_prob', type=float, default=0.2)

    args = parser.parse_args()

    return args

def scratch_t3(parser):

    parser.add_argument('--is_crop_topview', type=int, default=0)
    parser.add_argument('--is_draw_centerlines', type=int, default=0)
    parser.add_argument('--is_best_lane_only', type=int, default=0)
    parser.add_argument('--is_random_path_order', type=int, default=1)
    parser.add_argument('--is_repeat_scene', type=int, default=0)
    parser.add_argument('--is_train_dis', type=int, default=1)

    # ----------------------------------------
    #        | value | description
    # type 0 |   0   | neighbors close to a LP
    # type 1 |   1   | whole neighbors
    # type 2 |   2   | the closest to target agent & close to a LP

    parser.add_argument('--social_pooling_type', type=int, default=0)

    parser.add_argument('--x_crop_min', type=float, default=-20)
    parser.add_argument('--x_crop_max', type=float, default=80)
    parser.add_argument('--y_crop_min', type=float, default=-50)
    parser.add_argument('--y_crop_max', type=float, default=50)

    parser.add_argument('--path_resol', type=float, default=1.0)
    parser.add_argument('--max_path_len_forward', type=float, default=80)
    parser.add_argument('--max_path_len_backward', type=float, default=10)
    parser.add_argument('--local_path_len', type=float, default=30)
    parser.add_argument('--local_path_step', type=float, default=10)
    parser.add_argument('--lp_ngh_dist_thr', type=float, default=4.5)

    parser.add_argument('--num_max_paths', type=int, default=10)
    parser.add_argument('--lane_feat_dim', type=int, default=64)

    parser.add_argument('--pos_emb_dim', type=int, default=16)
    parser.add_argument('--rel_emb_dim', type=int, default=32)
    parser.add_argument('--traj_enc_h_dim', type=int, default=16)
    parser.add_argument('--sce_enc_dim', type=int, default=256)
    parser.add_argument('--traj_dec_h_dim', type=int, default=128)

    parser.add_argument('--gan_prior_prob', type=float, default=0.5)

    parser.add_argument('--z_dim', type=int, default=16)

    args = parser.parse_args()

    return args

def baseline(parser):

    parser.add_argument('--is_crop_topview', type=int, default=0)
    parser.add_argument('--is_draw_centerlines', type=int, default=0)
    parser.add_argument('--is_best_lane_only', type=int, default=1)
    parser.add_argument('--is_random_path_order', type=int, default=1)
    parser.add_argument('--is_repeat_scene', type=int, default=0)
    parser.add_argument('--is_bos_obj', type=int, default=0)

    parser.add_argument('--x_crop_min', type=float, default=-20)
    parser.add_argument('--x_crop_max', type=float, default=80)
    parser.add_argument('--y_crop_min', type=float, default=-50)
    parser.add_argument('--y_crop_max', type=float, default=50)

    parser.add_argument('--path_resol', type=float, default=1.0)
    parser.add_argument('--max_path_len_forward', type=float, default=80)
    parser.add_argument('--max_path_len_backward', type=float, default=10)
    parser.add_argument('--local_path_len', type=float, default=30)
    parser.add_argument('--local_path_step', type=float, default=10)

    parser.add_argument('--num_max_paths', type=int, default=10)
    parser.add_argument('--lane_feat_dim', type=int, default=64)

    parser.add_argument('--pos_emb_dim', type=int, default=16)
    parser.add_argument('--rel_emb_dim', type=int, default=32)
    parser.add_argument('--traj_enc_h_dim', type=int, default=16)
    parser.add_argument('--sce_enc_dim', type=int, default=256)
    parser.add_argument('--traj_dec_h_dim', type=int, default=128)

    parser.add_argument('--z_dim', type=int, default=16)

    args = parser.parse_args()

    return args

def baseline_nf(parser):

    parser.add_argument('--is_crop_topview', type=int, default=0)
    parser.add_argument('--is_draw_centerlines', type=int, default=0)
    parser.add_argument('--is_best_lane_only', type=int, default=1)
    parser.add_argument('--is_random_path_order', type=int, default=1)
    parser.add_argument('--is_repeat_scene', type=int, default=0)
    parser.add_argument('--is_bos_obj', type=int, default=0)

    parser.add_argument('--x_crop_min', type=float, default=-20)
    parser.add_argument('--x_crop_max', type=float, default=80)
    parser.add_argument('--y_crop_min', type=float, default=-50)
    parser.add_argument('--y_crop_max', type=float, default=50)

    parser.add_argument('--path_resol', type=float, default=1.0)
    parser.add_argument('--max_path_len_forward', type=float, default=80)
    parser.add_argument('--max_path_len_backward', type=float, default=10)
    parser.add_argument('--local_path_len', type=float, default=30)
    parser.add_argument('--local_path_step', type=float, default=10)

    parser.add_argument('--num_max_paths', type=int, default=10)
    parser.add_argument('--lane_feat_dim', type=int, default=64)

    parser.add_argument('--pos_emb_dim', type=int, default=16)
    parser.add_argument('--rel_emb_dim', type=int, default=32)
    parser.add_argument('--traj_enc_h_dim', type=int, default=16)
    parser.add_argument('--sce_enc_dim', type=int, default=256)
    parser.add_argument('--traj_dec_h_dim', type=int, default=128)

    parser.add_argument('--z_dim', type=int, default=16)

    args = parser.parse_args()

    return args

def NMP(parser):
    parser.add_argument('--eval_method', type=int, default=0) # 0 : minADE, 1 : minCost
    parser.add_argument('--traj_set_eps', type=int, default=2)
    parser.add_argument('--neg_traj_ratio', type=float, default=0.66)
    parser.add_argument('--collision_dist', type=float, default=4.0)


    parser.add_argument('--is_crop_topview', type=int, default=0)
    parser.add_argument('--x_crop_min', type=float, default=-20)
    parser.add_argument('--x_crop_max', type=float, default=80)
    parser.add_argument('--y_crop_min', type=float, default=-50)
    parser.add_argument('--y_crop_max', type=float, default=50)

    parser.add_argument('--pos_emb_dim', type=int, default=16)
    parser.add_argument('--rel_emb_dim', type=int, default=32)
    parser.add_argument('--traj_enc_h_dim', type=int, default=16)
    parser.add_argument('--sce_enc_dim', type=int, default=256)
    parser.add_argument('--traj_dec_h_dim', type=int, default=128)

    parser.add_argument('--cnn_outch_dim', type=float, default=128)

    args = parser.parse_args()

    return args

def agentformer(parser):

    parser.add_argument('--target_dset', type=str, default='eth')
    parser.add_argument('--step', type=int, default=10)
    parser.add_argument('--lr_scheduler_gamma', type=float, default=0.4)

    parser.add_argument('--obs_len', type=int, default=8)
    parser.add_argument('--pred_len', type=int, default=12)
    parser.add_argument('--min_obs_len', type=int, default=8)
    parser.add_argument('--traj_scale', type=float, default=2.0)

    parser.add_argument('--model_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)

    # context encoder/future encoder/future decoder
    parser.add_argument('--pos_concat', type=bool, default=True)
    parser.add_argument('--use_agent_enc', type=bool, default=False)
    parser.add_argument('--max_agent_len', type=int, default=128)
    parser.add_argument('--nlayer', type=int, default=2)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--ff_dim', type=int, default=512)
    parser.add_argument('--nz', type=int, default=32)
    parser.add_argument('--pooling', type=str, default='mean')


    args = parser.parse_args()

    return args