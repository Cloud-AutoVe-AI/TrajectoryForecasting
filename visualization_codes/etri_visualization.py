from ETRIDataset.visualization import Visualizer
from utils.functions import *
from helper import load_datasetloader, load_solvers

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_id', type=int, default=1331)
    parser.add_argument('--gpu_num', type=int, default=0)
    parser.add_argument('--model_name', type=str, default='autove')
    parser.add_argument('--start_frm_idx', type=int, default=0)
    parser.add_argument('--best_k', type=int, default=10)
    parser.add_argument('--map_size', type=int, default=1024)
    parser.add_argument('--t_skip', type=int, default=1)
    parser.add_argument('--scene_range', type=float, default=60)
    parser.add_argument('--is_save', type=int, default=0) # update, 220216
    parser.add_argument('--is_target_only', type=int, default=1)  # update, 220216
    args = parser.parse_args()
    test(args)

def return_target_scenes():

    # update, 220217
    target_scenes = []

    # logid : 0075
    target_scenes += [i for i in range(0, 4+1)]
    target_scenes += [i for i in range(15, 19+1)]
    target_scenes += [i for i in range(180, 184+1)]
    target_scenes += [i for i in range(193, 197+1)]
    target_scenes += [i for i in range(200, 204+1)]

    # logid : 0043
    target_scenes += [i for i in range(215, 219 + 1)]
    target_scenes += [i for i in range(240, 249 + 1)]
    target_scenes += [i for i in range(280, 289 + 1)]
    target_scenes += [i for i in range(350, 354 + 1)]
    target_scenes += [i for i in range(420, 424 + 1)]
    target_scenes += [i for i in range(450, 454 + 1)]
    target_scenes += [i for i in range(465, 469 + 1)]
    target_scenes += [i for i in range(480, 484 + 1)]
    target_scenes += [i for i in range(525, 529 + 1)]

    # logid : 0006
    target_scenes += [i for i in range(615, 619 + 1)]

    # logid : 0033
    target_scenes += [i for i in range(795, 799 + 1)]
    target_scenes += [i for i in range(840, 844 + 1)]

    # logid : 0016
    target_scenes += [i for i in range(970, 974 + 1)]
    target_scenes += [i for i in range(995, 999 + 1)]
    target_scenes += [i for i in range(1104, 1108 + 1)]
    # target_scenes += [i for i in range(1274, 1278 + 1)]
    target_scenes += [i for i in range(1300, 1304 + 1)]
    target_scenes += [i for i in range(1330, 1334 + 1)]

    # logid : 0110
    target_scenes += [i for i in range(1395, 1399 + 1)]
    target_scenes += [i for i in range(1410, 1414 + 1)]
    target_scenes += [i for i in range(1594, 1598 + 1)]

    return target_scenes


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
    saved_args.is_train_w_nuscenes = 0
    saved_args.limit_range = 50 # important ------
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

    # update, 220216
    folder_path = './Captures/capture_%d' % saved_args.exp_id
    if (args.is_save == 1):
        if (os.path.exists('./Captures') == False):
            os.mkdir('./Captures')

        if (os.path.exists(folder_path) == False):
            os.mkdir(folder_path)

    target_scenes = return_target_scenes()


    auto_play = False
    dataset_len = data_loader.num_test_scenes
    current_frame_idx = args.start_frm_idx
    while True:

        # update, 220217
        if (args.is_target_only == 1 and current_frame_idx not in target_scenes):
            current_frame_idx += 1
            if (current_frame_idx == dataset_len):
                sys.exit()
            continue

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

        # draw point cloud topivew
        fig, ax = plt.subplots()
        img = 255 * np.ones(shape=(map_size, map_size, 3))
        ax.imshow(img.astype('float') / 255.0, extent=[0, map_size, 0, map_size])

        # draw hdmap
        ax = vs.topview_hdmap(ax, scene.agent_dict['EGO'].pose, x_range, y_range, map_size)

        # draw bbox
        num_agents = agent_ids.shape[0]
        for a in range(num_agents):
            a_token = scene.id_2_token_lookup[agent_ids[a]]
            agent = scene.agent_dict[a_token]
            if (agent.track_id == 'EGO'):
                continue
            ax = vs.topview_bbox(ax, agent, (0.5, 0.5, 0.5))

        # draw trajs
        if (saved_args.model_mode == 'pedestrian'):
            ego_traj = scene.agent_dict['EGO'].trajectory[:, 1:3]
            ax = vs.topview_trajectory(ax, ego_traj, [])

        num_agents = agent_ids.shape[0]
        for a in range(num_agents):

            if (a == 0):
                continue

            a_token = scene.id_2_token_lookup[agent_ids[a]]
            agent = scene.agent_dict[a_token]

            if (agent.track_id != 'EGO'):
                ax = vs.topview_bbox(ax, agent, (1, 0, 0))

            gt_traj = overall_traj[:, a, :]
            for k in range(args.best_k):
                est_traj = pred_trajs_valid[k, :, a, :]
                ax = vs.topview_trajectory(ax, gt_traj, est_traj)

        plt.axis([0, map_size, 0, map_size])
        # plt.show()

        img = vs.fig_to_nparray(fig, ax)
        text = '[Log ID %s, Scene # %d]' % (scene.log_token, current_frame_idx)
        cv2.putText(img, text, (20, 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255))


        # update, 220216
        if (args.is_save == 1):
            file_name = 'img_%04d.png' % (current_frame_idx)
            cv2.imwrite(os.path.join(folder_path, file_name), img)
            current_frame_idx += 1
            if (current_frame_idx == dataset_len):
                sys.exit()

        else:
            # show image
            cv2.imshow('', img)

            # key actions
            if auto_play:
                k = cv2.waitKey(1) & 0xFF
                current_frame_idx = current_frame_idx + 1
            else:
                k = cv2.waitKey(0) & 0xFF

            if k == 27:  # esc key
                cv2.destroyAllWindows()
                return False
            elif k == 32:  # Space key
                if auto_play:
                    auto_play = False
                else:
                    auto_play = True

            if k == ord('e'):  # jump actions
                if current_frame_idx + 10 < dataset_len:
                    current_frame_idx = current_frame_idx + 10
                else:
                    current_frame_idx = dataset_len - 1
            if k == ord('q'):
                if current_frame_idx - 10 >= 0:
                    current_frame_idx = current_frame_idx - 10
                else:
                    current_frame_idx = 0
            if k == ord('/'):
                current_frame_idx = 0

            if not auto_play:
                if k == 83 or k == ord('d'):  # '->' key
                    if not current_frame_idx >= dataset_len:
                        current_frame_idx = current_frame_idx + 1
                elif k == 81 or k == ord('a'):  # '<-' key
                    if current_frame_idx - 1 >= 0:
                        current_frame_idx = current_frame_idx - 1

            if current_frame_idx >= dataset_len:
                current_frame_idx = dataset_len - 1

if __name__ == '__main__':
    main()


