from ETRIDataset.visualization import Visualizer
from utils.functions import *
from helper import load_datasetloader, load_solvers

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_id', type=int, default=1025)
    parser.add_argument('--gpu_num', type=int, default=0)
    parser.add_argument('--model_name', type=str, default='autove')
    parser.add_argument('--start_frm_idx', type=int, default=221)
    parser.add_argument('--best_k', type=int, default=10)
    parser.add_argument('--map_size', type=int, default=1024)
    parser.add_argument('--t_skip', type=int, default=1)
    parser.add_argument('--scene_range', type=float, default=60)

    args = parser.parse_args()
    test(args)

def test(args):


    # parent of working directory is base
    abspath = os.path.dirname(os.path.realpath(__file__))
    os.chdir(Path(abspath).parent.absolute())

    # CUDA setting
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(args.gpu_num))

    # type definition
    long_dtype, float_dtype = get_dtypes(useGPU=True)

    folder_name = 'ETRI_' + args.model_name + '_model' + str(args.exp_id)
    path = os.path.join('./saved_models/', folder_name)

    # load parameter setting
    with open(os.path.join(path, 'config.pkl'), 'rb') as f:
        saved_args = pickle.load(f)
    saved_args.best_k = args.best_k
    saved_args.is_train_w_nuscenes = 0 # update, 211202
    saved_args.limit_range = 50
    saved_args.dataset_type = 'ETRI'
    print_training_info(saved_args)

    # load test data
    data_loader, _ = load_datasetloader(args=saved_args, isTrain=False, dtype=torch.FloatTensor)

    # define network
    solver = load_solvers(saved_args, data_loader.num_test_scenes, float_dtype)
    ckp_idx = save_read_latest_checkpoint_num(os.path.join(solver.save_dir), 0, isSave=False)
    _ = solver.load_pretrained_network_params(ckp_idx)
    solver.mode_selection(isTrain=False)

    # evaluation setting
    t_skip = args.t_skip
    obs_len = int(saved_args.past_horizon_seconds * saved_args.target_sample_period)
    pred_len = int(saved_args.future_horizon_seconds * saved_args.target_sample_period)
    saved_args.best_k = args.best_k
    saved_args.batch_size = 1
    obs_len_ = obs_len

    # sub-sample trajs
    target_index_obs = np.array([-1 * _ for _ in range(0, obs_len, t_skip)])[::-1] + (obs_len-1)
    target_index_pred = np.array([_ for _ in range(t_skip - 1, pred_len, t_skip)])

    obs_len = len(target_index_obs)
    pred_len = len(target_index_pred)

    # scene range
    map_size = args.map_size
    x_range = (-1 * args.scene_range, args.scene_range)
    y_range = (-1 * args.scene_range, args.scene_range)
    z_range = (-3, 2)

    # visualizer
    vs = Visualizer(args=saved_args, map=data_loader.map,  x_range=x_range, y_range=y_range,
                    z_range=z_range, map_size=map_size, obs_len=obs_len, pred_len=pred_len)


    dataset_len = data_loader.num_test_scenes
    for current_frame_idx in range(args.start_frm_idx, dataset_len):

        # data loading
        data = data_loader.next_sample(current_frame_idx, mode='test')

        # inference
        obs_traj, future_traj, pred_trajs, agent_ids, scene, valid_scene_flag = solver.test(data, float_dtype, args.best_k)

        if (valid_scene_flag == False):
            current_frame_idx+=1
            continue

        # debug ---
        # vs.show_around_view_images(scene.sample_token)

        obs_traj_valid = obs_traj[target_index_obs, :, :]
        future_traj_valid = future_traj[target_index_pred, :, :]
        overall_traj = np.concatenate([obs_traj_valid, future_traj_valid], axis=0)
        pred_trajs_valid = pred_trajs[:, target_index_pred, :, :]


        num_agents = agent_ids.shape[0]
        for a in range(num_agents):

            # draw point cloud topivew
            fig, ax = plt.subplots()
            img = 255 * np.ones(shape=(map_size, map_size, 3))
            ax.imshow(img.astype('float') / 255.0, extent=[0, map_size, 0, map_size])

            # draw hdmap
            ax = vs.topview_hdmap(ax, scene.agent_dict['EGO'].pose, x_range, y_range, map_size)

            # draw bbox
            num_agents = agent_ids.shape[0]
            for n in range(num_agents):
                a_token = scene.id_2_token_lookup[agent_ids[n]]
                agent = scene.agent_dict[a_token]
                if (agent.track_id == 'EGO'):
                    continue

                if (a != n):
                    ax = vs.topview_bbox(ax, agent, (0.5, 0.5, 0.5))
                else:
                    ax = vs.topview_bbox(ax, agent, (1.0, 0.0, 0.0))

            # draw trajectory
            gt_traj = overall_traj[:, a, :]
            for k in range(args.best_k):
                est_traj = pred_trajs_valid[k, :, a, :]
                ax = vs.topview_trajectory(ax, gt_traj, est_traj)


            plt.axis([0, map_size, 0, map_size])
            img = vs.fig_to_nparray(fig, ax)
            text = '[Scene %d]' % current_frame_idx
            cv2.putText(img, text, (20, 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255))
            cv2.imshow('', img)
            cv2.waitKey(0)
            plt.close()



if __name__ == '__main__':
    main()


