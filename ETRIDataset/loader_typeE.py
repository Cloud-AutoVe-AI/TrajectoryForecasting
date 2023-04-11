from utils.libraries import *
from ETRIDataset.preprocess import DatasetBuilder
from torch.utils.data import Dataset
from utils.functions import read_config
from ETRIDataset.visualization import *
from ETRIDataset.map import Map

class DatasetLoader(Dataset):

    def __init__(self, args, dtype, isTrain=True):

        '''
        For training vehicle model
        '''

        # exp settings
        exp_type = 'train' if isTrain else 'test'

        # nuscenes map api
        self.map = Map(args)

        # params
        config = read_config()

        self.dtype = dtype
        self.target_sample_period = args.target_sample_period
        self.obs_len = int(args.past_horizon_seconds * args.target_sample_period)
        self.pred_len = int(args.future_horizon_seconds * args.target_sample_period)
        self.min_obs_len = int(args.min_past_horizon_seconds * args.target_sample_period)
        self.sub_step = int(10.0 / args.target_sample_period)

        self.limit_range = args.limit_range
        self.x_range = (args.x_range_min, args.x_range_max)
        self.y_range = (args.y_range_min, args.y_range_max)
        self.z_range = (args.z_range_min, args.z_range_max)
        self.map_size = args.map_size
        self.lidar_map_ch_dim = args.lidar_map_ch_dim
        self.num_lidar_sweeps = args.num_lidar_sweeps

        self.use_preprocessed_lidar = args.use_preprocessed_lidar
        self.voxel_save_dir = config['ETRI']['voxel_dir']

        self.stop_agents_remove_prob = args.stop_agents_remove_prob
        self.limit_range_change_prob = args.limit_range_change_prob
        self.isAug = True if (args.is_random_rotate == 1) else False

        # checks existance of dataset file and create
        save_path = config['ETRI']['preproc_dataset_path']
        if (os.path.exists(save_path)==False):
            os.mkdir(save_path)
        file_name = 'etri_%s.cpkl' % exp_type

        builder = DatasetBuilder(args, map=0, isTrain=isTrain)
        if (os.path.exists(os.path.join(save_path, file_name))==False):
            builder.make_preprocessed_data(os.path.join(save_path, file_name), exp_type)

        # load dataset file
        with open(os.path.join(save_path, file_name), 'rb') as f:
            dataset = dill.load(f, encoding='latin1')
            print(">> {%s} is loaded .." % (os.path.join(save_path, file_name)))

        self.train_data = dataset[0]
        self.valid_data = dataset[1]
        self.test_data = dataset[2]


        # load preprocessed nuscenes dataset
        self.train_data_flag = np.ones(shape=(len(self.train_data)))
        self.valid_data_flag = np.ones(shape=(len(self.valid_data)))
        self.test_data_flag = np.ones(shape=(len(self.test_data)))

        # un-available now ------------------------------------
        self.is_train_w_nuscenes = args.is_train_w_nuscenes
        if (self.is_train_w_nuscenes == 1 and isTrain):
            from utils.dataset_conversion import Dtype_Scene as Scene
            n_save_path = '/home/dooseop/DATASET/voss/nuscenes'
            for part in range(10):
                n_file_name = 'nuscenes_typeDconv_%ds%ds_part%d_ped.cpkl' % (
                args.past_horizon_seconds, args.future_horizon_seconds, part)
                try:
                    with open(os.path.join(n_save_path, n_file_name), 'rb') as f:
                        n_conv_data = dill.load(f, encoding='latin1')
                        print(">> {%s} is loaded .." % (os.path.join(n_save_path, n_file_name)))

                    num_scenes = len(n_conv_data)
                    ntrain_data = n_conv_data[:int(num_scenes * 0.8)]
                    nvalid_data = n_conv_data[int(num_scenes * 0.8):int(num_scenes * 0.85)]

                    self.train_data_flag = np.concatenate([self.train_data_flag, np.zeros(shape=(len(ntrain_data)))])
                    self.valid_data_flag = np.concatenate([self.valid_data_flag, np.zeros(shape=(len(nvalid_data)))])
                    self.train_data += ntrain_data
                    self.valid_data += nvalid_data
                except:
                    print(">> [warning] unable to load {%s}" % n_file_name)
        # un-available now ------------------------------------

        # current dataset info
        self.num_train_scenes = len(self.train_data)
        self.num_valid_scenes = len(self.valid_data)
        self.num_test_scenes = len(self.test_data)
        print(">> Loader is loaded from {%s} " % os.path.basename(__file__))


    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):

        # current scene data
        scene = self.train_data[idx]
        flag = self.train_data_flag[idx]
        if (flag == 1):
            data = self.extract_data_from_scene(scene, isTrain=True, isAug=self.isAug)
            if (len(data) > 0):
                obs_traj, future_traj, obs_traj_e, future_traj_e, feature_topview, R_map, R_traj, num_agents, agent_id = data

                obs_traj = torch.from_numpy(obs_traj).type(self.dtype)
                future_traj = torch.from_numpy(future_traj).type(self.dtype)
                obs_traj_e = torch.from_numpy(obs_traj_e).type(self.dtype)
                future_traj_e = torch.from_numpy(future_traj_e).type(self.dtype)
                feature_topview = torch.from_numpy(feature_topview).permute(2, 0, 1).type(self.dtype)
                feature_topview = torch.unsqueeze(feature_topview, dim=0)
                R_map = torch.from_numpy(R_map).type(self.dtype)
                R_traj = torch.from_numpy(R_traj).type(self.dtype)
            else:
                obs_traj = torch.from_numpy(np.full(shape=(self.obs_len, 1, 3), fill_value=np.nan)).type(self.dtype)
                future_traj = torch.from_numpy(np.full(shape=(self.pred_len, 1, 3), fill_value=np.nan)).type(self.dtype)
                obs_traj_e = torch.from_numpy(np.full(shape=(self.obs_len, 1, 2), fill_value=np.nan)).type(self.dtype)
                future_traj_e = torch.from_numpy(np.full(shape=(self.pred_len, 1, 2), fill_value=np.nan)).type(self.dtype)
                feature_topview = torch.from_numpy(np.full(shape=(self.map_size, self.map_size, 4), fill_value=np.nan)).permute(2, 0, 1).type(self.dtype)
                feature_topview = torch.unsqueeze(feature_topview, dim=0)
                R_map = torch.from_numpy(np.full(shape=(1, 2, 2), fill_value=np.nan)).type(self.dtype)
                R_traj = torch.from_numpy(np.full(shape=(1, 2, 2), fill_value=np.nan)).type(self.dtype)
                num_agents = 1
        else:
            feature_topview = scene.feature_topview.astype('float') / 255.0
            num_agents = scene.num_agents

            obs_traj = torch.from_numpy(scene.obs_traj).type(self.dtype)
            future_traj = torch.from_numpy(scene.future_traj).type(self.dtype)
            obs_traj_e = torch.from_numpy(scene.obs_traj_a).type(self.dtype)
            future_traj_e = torch.from_numpy(scene.future_traj_a).type(self.dtype)
            feature_topview = torch.from_numpy(feature_topview).permute(2, 0, 1).type(self.dtype)
            feature_topview = torch.unsqueeze(feature_topview, dim=0)
            R_map = torch.from_numpy(scene.R_map).type(self.dtype)
            R_traj = torch.from_numpy(scene.R_traj).type(self.dtype)

        return obs_traj, future_traj, obs_traj_e, future_traj_e, feature_topview, R_map, R_traj, num_agents

    def next_sample(self, index, mode):

        if (mode == 'valid'):
            scene = self.valid_data[index]
            flag = self.valid_data_flag[index]
        else:
            scene = self.test_data[index]
            flag = 1

        if (flag == 1):
            data = self.extract_data_from_scene(scene, isTrain=False)
            if (len(data) > 0):
                obs_traj, future_traj, obs_traj_e, future_traj_e, feature_topview, R_map, R_traj, num_agents, agent_id = data

                # debug -------------
                # img = (255.0 * np.copy(feature_topview[:, :, :3])).astype('uint8')
                #
                # overall_trajs = np.concatenate([obs_traj[:, :, :2], future_traj[:, :, :2]], axis=0)
                # for _, id in enumerate(agent_id[0, :]):
                #     agent_key = scene.id_2_token_lookup[int(id)]
                #     agent = scene.agent_dict[agent_key]
                #
                #     if (agent.agent_id in agent_id[0, :].tolist()):
                #
                #         img = draw_traj_on_topview(img, overall_trajs[:, np.argwhere(agent.agent_id == agent_id[0]).item(), :2],
                #                                    self.obs_len, self.x_range, self.y_range, self.map_size, (0, 255, 255))
                #
                # cv2.imshow("", img)
                # cv2.waitKey(0)
                #
                # grid_size = 0.1
                # num_grid = 400
                # x = create_ROI_ped(grid_size, num_grid).reshape(num_grid ** 2, 2)
                # for a in range(0, obs_traj.shape[1]):
                #
                #     a_id = agent_id[0, a]
                #     a_token = scene.id_2_token_lookup[a_id]
                #     agent = scene.agent_dict[a_token]
                #
                #     cur_pos = obs_traj[-1, a, :2].reshape(1, 2)
                #     prev_pos = obs_traj[-2, a, :2].reshape(1, 2)
                #     cur_R = R_map[a]
                #
                #     x_rot = np.matmul(cur_R, x.T).T + cur_pos
                #     aa = pooling_operation(img, x_rot, self.x_range, self.y_range, self.map_size)
                #     b = aa.reshape(num_grid, num_grid, 3).astype('uint8')
                #     b = cv2.resize(b, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                #
                #     obs_traj_ego = np.matmul(R_traj[a], (overall_trajs[:, a, :2]-cur_pos).T).T
                #     c = np.zeros(shape=(2*num_grid, 2*num_grid, 3))
                #     c = draw_traj_on_topview(c, obs_traj_ego, self.obs_len, (-40, 40), (-40, 40), 2*num_grid, (0, 255, 255))
                #
                #     d = np.hstack([b, c])
                #     cv2.imshow('test', d.astype('uint8'))
                #     cv2.waitKey(0)
                # debug ------------
            else:
                obs_traj = np.full(shape=(self.obs_len, 1, 3), fill_value=np.nan)
                future_traj = np.full(shape=(self.pred_len, 1, 3), fill_value=np.nan)
                obs_traj_e = np.full(shape=(self.obs_len, 1, 2), fill_value=np.nan)
                future_traj_e = np.full(shape=(self.pred_len, 1, 2), fill_value=np.nan)
                feature_topview = np.full(shape=(self.map_size, self.map_size, 4), fill_value=np.nan)
                R_map = np.full(shape=(1, 2, 2), fill_value=np.nan)
                R_traj = np.full(shape=(1, 2, 2), fill_value=np.nan)
                agent_id = np.ones(shape=(1, 1))
                num_agents = 1
        else:
            obs_traj = scene.obs_traj
            future_traj = scene.future_traj
            obs_traj_e = scene.obs_traj_a
            future_traj_e = scene.future_traj_a
            feature_topview = scene.feature_topview.astype('float') / 255.0
            R_map = scene.R_map
            R_traj = scene.R_traj
            num_agents = scene.num_agents
            agent_id = scene.agent_id

        return obs_traj, future_traj, obs_traj_e, future_traj_e, feature_topview, R_map, R_traj, num_agents, agent_id, scene

    def extract_data_from_scene(self, scene, isTrain=True, isAug=False):

        # trajectory ---------
        num_total_agents = len(scene.agent_dict)
        agent_id = np.zeros(shape=(1, num_total_agents))
        category = np.zeros(shape=(1, num_total_agents)) # 0: vehicle, 1:pedestrian
        trajectories = np.full(shape=(self.obs_len + self.pred_len, num_total_agents, 3), fill_value=np.nan)
        track_ids = []
        for idx, track_id in enumerate(scene.agent_dict):
            agent_id[0, idx] = scene.agent_dict[track_id].agent_id
            track_ids.append(track_id)
            if (scene.agent_dict[track_id].type == 'vehicle'):
                category[0, idx] = 0
            else:
                category[0, idx] = 1

            trajectory = scene.agent_dict[track_id].trajectory
            trajectories[:, idx, :] = trajectory[:, 1:]

        # debug ---
        track_ids = np.array(track_ids).reshape(1, num_total_agents)
        # debug ---

        # find agents inside the limit range
        limit_range = self.limit_range
        if (np.random.rand(1) < self.limit_range_change_prob and isTrain):
            limit_range = float(random.randint(15, int(self.limit_range)))

        valid_flag = np.sqrt(np.sum(trajectories[self.obs_len - 1, :, :2] ** 2, axis=1)) < limit_range
        trajectories = trajectories[:, valid_flag, :]
        agent_id = agent_id[:, valid_flag]
        track_ids = track_ids[:, valid_flag]
        category = category[:, valid_flag]


        # find agents who has valid observation
        valid_flag = np.sum(trajectories[self.obs_len - self.min_obs_len:self.obs_len, :, 0], axis=0) > -1000
        trajectories = trajectories[:, valid_flag, :]
        agent_id = agent_id[:, valid_flag]
        track_ids = track_ids[:, valid_flag]
        category = category[:, valid_flag]


        # find trajectories that have full future traj,
        valid_flag = np.min(trajectories[self.obs_len:, :, 0], axis=0) > -1000
        trajectories = trajectories[:, valid_flag, :]
        agent_id = agent_id[:, valid_flag]
        track_ids = track_ids[:, valid_flag]
        category = category[:, valid_flag]


        # pedestrian only flag
        valid_flag = (category[0, :] == 1)
        if (np.count_nonzero(valid_flag) == 0):
            return []


        # random rotation augmentation
        degree = None
        if (isTrain and isAug):
            num_agents = trajectories.shape[1]
            trajectories_z = np.copy(trajectories[:, :, 2]).reshape(self.obs_len + self.pred_len, num_agents, 1)
            trajectories, degree = random_rotation_augmentation(trajectories)
            trajectories = np.concatenate([trajectories, trajectories_z], axis=2)

        # topview img
        feature_topview = self.map.make_topview_map_typeE(scene.agent_dict['EGO'].pose, self.x_range, self.y_range,
                                                    self.map_size, trajectories[:self.obs_len], category)

        if (degree is not None):
            img = (255.0 * np.copy(feature_topview)).astype('uint8')
            height, width, channel = img.shape
            matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 360-degree, 1)
            feature_topview = cv2.warpAffine(img, matrix, (width, height)).astype('float') / 255.0


        # pedestrian only flag
        trajectories = trajectories[:, valid_flag, :]
        agent_id = agent_id[:, valid_flag]
        track_ids = track_ids[:, valid_flag]
        num_agents = trajectories.shape[1]

        # split into observation and future
        obs_traj = trajectories[:self.obs_len]
        future_traj = trajectories[self.obs_len:]

        # rotation matrix, note : pedestrian has eye(2) rotation matrix)
        R_map = np.concatenate([np.expand_dims(np.eye(2), axis=0) for _ in range(num_agents)], axis=0)
        R_traj = np.copy(R_map)

        # 'nan' in ego-centric trajectories becomes zero
        nan_pos_obs = np.isnan(obs_traj[:, :, 2])
        nan_pos_future = np.isnan(future_traj[:, :, 2])

        # conversion to agent-centric
        obs_traj_e = []
        future_traj_e = []
        for a in range(num_agents):
            cur_R = R_traj[a]  # 2 x 2
            cur_center = obs_traj[-1, a, :2].reshape(1, 2)  # 1 x 2
            cur_obs_traj = obs_traj[:, a, :2]  # seq_len x 2
            cur_future_traj = future_traj[:, a, :2]

            cur_obs_traj_e = np.matmul(cur_R, (cur_obs_traj - cur_center).T).T  # seq_len x 2
            cur_future_traj_e = np.matmul(cur_R, (cur_future_traj - cur_center).T).T

            obs_traj_e.append(cur_obs_traj_e.reshape(self.obs_len, 1, 2))
            future_traj_e.append(cur_future_traj_e.reshape(self.pred_len, 1, 2))

        obs_traj_e = np.concatenate(obs_traj_e, axis=1)
        future_traj_e = np.concatenate(future_traj_e, axis=1)

        obs_traj_e[nan_pos_obs] = 0
        future_traj_e[nan_pos_future] = 0

        return [obs_traj, future_traj, obs_traj_e, future_traj_e, feature_topview,
                R_map, R_traj, num_agents, agent_id]


def rotate_around_point(xy, degree, origin=(0, 0)):

    radians = math.radians(degree)
    x, y = xy
    offset_x, offset_y = origin
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = math.cos(radians)
    sin_rad = math.sin(radians)
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y

    return qx, qy

def random_rotation_augmentation(traj_data):

    seq_len, num_vids, _ = traj_data.shape
    traj_data_rt = []

    degree = random.randint(0, 359)
    for v in range(num_vids):

        x0 = np.copy(traj_data[:, v, 0])
        y0 = np.copy(traj_data[:, v, 1])

        xr, yr = rotate_around_point((x0, y0), degree, origin=(0, 0))

        xyr = np.concatenate([xr.reshape(seq_len, 1), yr.reshape(seq_len, 1)], axis=1)
        xyr = np.around(xyr, decimals=4)
        traj_data_rt.append(np.expand_dims(xyr, axis=1))

    if (num_vids == 1):
        return traj_data_rt[0], degree
    else:
        return np.concatenate(traj_data_rt, axis=1), degree