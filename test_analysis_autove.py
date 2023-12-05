import numpy as np

from utils.functions import *
from utils.loss import calc_ED_error
from helper import load_datasetloader, load_solvers

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_id', type=int, default=1075)
    parser.add_argument('--gpu_num', type=int, default=0)
    parser.add_argument('--dataset_type', type=str, default='ETRI')
    parser.add_argument('--model_name', type=str, default='autove')
    parser.add_argument('--step_size', type=int, default=1)
    parser.add_argument('--best_k', type=int, default=15)
    parser.add_argument('--is_test_all', type=int, default=1)
    parser.add_argument('--model_num', type=int, default=36)

    args = parser.parse_args()
    test(args)

    a = 3

def test(args):


    # CUDA setting
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(args.gpu_num))

    # type definition
    _, float_dtype = get_dtypes(useGPU=True)

    # path to saved network
    folder_name = args.dataset_type + '_' + args.model_name + '_model' + str(args.exp_id)
    path = os.path.join('./saved_models/', folder_name)

    # load parameter setting
    with open(os.path.join(path, 'config.pkl'), 'rb') as f:
        saved_args = pickle.load(f)
    saved_args.best_k = args.best_k

    # update, 220217
    saved_args.scene_accept_prob = 1.0
    if ('scratch' in args.model_name or 'baseline' in args.model_name):
        saved_args.limit_range = 200
    elif ('autove' in args.model_name):
        saved_args.limit_range = 50

    # load test data
    data_loader, _ = load_datasetloader(args=saved_args, isTrain=False, dtype=torch.FloatTensor)

    # define network
    solver = load_solvers(saved_args, data_loader.num_test_scenes, float_dtype)
    ckp_idx_list = read_all_saved_param_idx(solver.save_dir)

    target_models = []
    if (args.is_test_all == 0):
        target_models.append(args.model_num)
    else:
        target_models += ckp_idx_list


    for _, ckp_id in enumerate(ckp_idx_list):

        if (ckp_id not in target_models):
            print(">> [SKIP] current model %d is not in target model list!" % ckp_id)
            continue

        solver.load_pretrained_network_params(ckp_id)
        solver.mode_selection(isTrain=False)

        ADE1, FDE1, ADE10, FDE10, ADEK, FDEK = [], [], [], [], [], []
        ADE1_high_curvature, ADE1_mid_curvature, ADE1_low_curvature = [], [], []
        ADE10_high_curvature, ADE10_mid_curvature, ADE10_low_curvature = [], [], []
        ADE1_high_speed, ADE1_mid_speed, ADE1_low_speed = [], [], []
        ADE10_high_speed, ADE10_mid_speed, ADE10_low_speed = [], [], []
        for b in range(0, len(data_loader.test_data), args.step_size):

            # data loading
            data = data_loader.next_sample(b, mode='test')

            # inference
            obs_traj, future_traj, pred_trajs, agent_ids, scene, valid_scene_flag = solver.test(data, float_dtype, args.best_k)

            if (valid_scene_flag == False):
                continue

            num_agents = pred_trajs.shape[2]
            for o in range(num_agents):

                # debug --------------
                target_traj = np.concatenate([obs_traj[:, o, :2], future_traj[:, o, :2]], axis=0)

                # calc speed
                disp = (target_traj[1:] - target_traj[:-1])[:, :2]
                max_speeds_kmph = np.max(3.6 * saved_args.target_sample_period * np.sqrt(np.sum(disp ** 2, axis=1)))

                # calc curvature
                path_len = np.sum(np.sqrt(np.sum(disp ** 2, axis=1)))
                path_dist = np.sqrt(np.sum((target_traj[0] - target_traj[-1]) ** 2))
                cur_curvature = path_len / (path_dist + 1e-10)

                if (max_speeds_kmph < 10):
                    cur_curvature = 1.00001
                # debug --------------

                ADE_k, FDE_k = [], []
                for k in range(args.best_k):
                    error = np.sqrt(np.sum((pred_trajs[k, :, o, :2] - future_traj[:, o, :2])**2, axis=1))
                    ADE_k.append(np.mean(error))
                    FDE_k.append(np.mean(error[-1]))

                error_ADE, error_FDE = calc_ED_error(o, 1, args.best_k, pred_trajs, future_traj, ADE_k, FDE_k)
                ADE1.append(error_ADE)
                FDE1.append(error_FDE)

                # debug --------------
                if (max_speeds_kmph < 20):
                    ADE1_low_speed.append(error_ADE)
                elif (max_speeds_kmph >= 20 and max_speeds_kmph < 50):
                    ADE1_mid_speed.append(error_ADE)
                else:
                    ADE1_high_speed.append(error_ADE)

                if (cur_curvature < 1.001):
                    ADE1_low_curvature.append(error_ADE)
                elif (cur_curvature >= 1.001 and cur_curvature < 1.05):
                    ADE1_mid_curvature.append(error_ADE)
                else:
                    ADE1_high_curvature.append(error_ADE)
                # debug --------------


                error_ADE, error_FDE = calc_ED_error(o, 10, args.best_k, pred_trajs, future_traj, ADE_k, FDE_k)
                ADE10.append(error_ADE)
                FDE10.append(error_FDE)

                # debug --------------
                if (max_speeds_kmph < 20):
                    ADE10_low_speed.append(error_ADE)
                elif (max_speeds_kmph >= 20 and max_speeds_kmph < 50):
                    ADE10_mid_speed.append(error_ADE)
                else:
                    ADE10_high_speed.append(error_ADE)

                if (cur_curvature < 1.001):
                    ADE10_low_curvature.append(error_ADE)
                elif (cur_curvature >= 1.001 and cur_curvature < 1.05):
                    ADE10_mid_curvature.append(error_ADE)
                else:
                    ADE10_high_curvature.append(error_ADE)
                # debug --------------

                error_ADE, error_FDE = calc_ED_error(o, args.best_k, args.best_k, pred_trajs, future_traj, ADE_k, FDE_k)
                ADEK.append(error_ADE)
                FDEK.append(error_FDE)

            print_current_test_progress(b, len(data_loader.test_data))

        print('--------------------------------------------------------------')
        print(">> ADE1 : %.4f, FDE1 : %.4f" % (np.mean(ADE1), np.mean(FDE1)))
        print(">> ADE10 : %.4f, FDE10 : %.4f" % (np.mean(ADE10), np.mean(FDE10)))
        print(">> ADE%d : %.4f, FDE%d : %.4f" % (args.best_k, np.mean(ADEK), args.best_k, np.mean(FDEK)))
        print('--------------------------------------------------------------')
        print(">> low speed - ADE1 : %.4f, ADE10 : %.4f" % (np.mean(ADE1_low_speed), np.mean(ADE10_low_speed)))
        print(">> mid speed - ADE1 : %.4f, ADE10 : %.4f" % (np.mean(ADE1_mid_speed), np.mean(ADE10_mid_speed)))
        print(">> high speed - ADE1 : %.4f, ADE10 : %.4f" % (np.mean(ADE1_high_speed), np.mean(ADE10_high_speed)))
        print('--------------------------------------------------------------')
        print(">> low curvature - ADE1 : %.4f, ADE10 : %.4f" % (np.mean(ADE1_low_curvature), np.mean(ADE10_low_curvature)))
        print(">> mid curvature - ADE1 : %.4f, ADE10 : %.4f" % (np.mean(ADE1_mid_curvature), np.mean(ADE10_mid_curvature)))
        print(">> high curvature - ADE1 : %.4f, ADE10 : %.4f" % (np.mean(ADE1_high_curvature), np.mean(ADE10_high_curvature)))
        print('--------------------------------------------------------------')



def is_moving(trajectory, seconds):

    '''
    trajectory : seq_len x 2
    '''
    moving_dist = np.sqrt(np.sum((trajectory[0] - trajectory[-1])**2))
    avg_speed = moving_dist / seconds

    if (avg_speed < 0.1):
        return False
    else:
        return True


if __name__ == '__main__':
    main()


