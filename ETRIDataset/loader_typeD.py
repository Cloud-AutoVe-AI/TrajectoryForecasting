import cv2

from utils.libraries import *
from ETRIDataset.preprocess import DatasetBuilder
from torch.utils.data import Dataset
from utils.functions import read_config
from ETRIDataset.visualization import *
from ETRIDataset.map import Map
from ETRIDataset.agent import Agent
from ETRIDataset.scene import Scene

class DatasetLoader(Dataset):

    def __init__(self, args, dtype, isTrain=True):

        '''
        For training vehicle model
        '''

        # exp settings
        exp_type = 'train' if isTrain else 'test'

        # read config
        config = read_config()

        # nuscenes map api
        self.map = Map(args)

        # parameters
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
        self.heading_speed_thr = args.heading_speed_thr

        self.add_heading_noise = args.add_heading_noise
        self.heading_noise_deg_std = args.heading_noise_deg_std
        self.is_bbox_heading = args.is_bbox_heading
        self.random_flip_prob = args.random_flip_prob
        self.agent_sampling_prob = args.agent_sampling_prob
        self.agent_sampling_thr = args.agent_sampling_thr
        self.pos_noise_prob = args.pos_noise_prob
        self.is_use_proc_traj = args.is_use_proc_traj
        self.pos_noise_std = args.pos_noise_std
        if (isTrain == False):
            self.is_use_proc_traj = 0

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

        self.train_data_flag = np.ones(shape=(len(self.train_data)))
        self.valid_data_flag = np.ones(shape=(len(self.valid_data)))
        self.test_data_flag = np.ones(shape=(len(self.test_data)))

        # load preprocessed nuscenes dataset
        self.is_train_w_nuscenes = args.is_train_w_nuscenes
        if (self.is_train_w_nuscenes == 1 and isTrain):
            from utils.dataset_conversion import Dtype_Scene as Scene
            n_save_path = '/home/dooseop/DATASET/voss/nuscenes'
            for part in range(10):
                n_file_name = 'nuscenes_typeDconv_%ds%ds_part%d.cpkl' % (args.past_horizon_seconds, args.future_horizon_seconds, part)
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

        # current dataset info
        self.num_train_scenes = len(self.train_data)
        self.num_valid_scenes = len(self.valid_data)
        self.num_test_scenes = len(self.test_data)
        print(">> Loader is loaded from {%s} " % os.path.basename(__file__))


    def __len__(self):
        return len(self.train_data)
        # return 32

    def __getitem__(self, idx):

        # current scene data
        scene = self.train_data[idx]
        flag = self.train_data_flag[idx]
        if (flag == 1):
            obs_traj, future_traj, obs_traj_e, future_traj_e, feature_topview, R_map, R_traj, Ht, Hb, num_agents, agent_id = \
                self.extract_data_from_scene(scene, isTrain=True)
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

        obs_traj = torch.from_numpy(obs_traj).type(self.dtype)
        future_traj = torch.from_numpy(future_traj).type(self.dtype)
        obs_traj_e = torch.from_numpy(obs_traj_e).type(self.dtype)
        future_traj_e = torch.from_numpy(future_traj_e).type(self.dtype)
        feature_topview = torch.from_numpy(feature_topview).permute(2, 0, 1).type(self.dtype)
        feature_topview = torch.unsqueeze(feature_topview, dim=0)
        R_map = torch.from_numpy(R_map).type(self.dtype)
        R_traj = torch.from_numpy(R_traj).type(self.dtype)

        return obs_traj, future_traj, obs_traj_e, future_traj_e, feature_topview, R_map, R_traj, num_agents

    def next_sample(self, index, mode):

        if (mode == 'valid'):
            scene = self.valid_data[index]
            flag = self.valid_data_flag[index]
        else:
            scene = self.test_data[index]
            # flag = self.test_data_flag[index]
            flag = 1

        if (flag == 1):
            obs_traj, future_traj, obs_traj_e, future_traj_e, feature_topview, R_map, R_traj, Ht, Hb, num_agents, agent_id = \
                self.extract_data_from_scene(scene, isTrain=False)

            # debug -------------
            # img = (255*feature_topview[:, :, :3]).astype('uint8')
            # overall_trajs = np.concatenate([obs_traj[:, :, :2], future_traj[:, :, :2]], axis=0)
            # for _, id in enumerate(agent_id[0, :]):
            #     agent_key = scene.id_2_token_lookup[int(id)]
            #     agent = scene.agent_dict[agent_key]
            #
            #     # if (agent.agent_id in agent_id[0, :].tolist()):
            #     if (id == 2):
            #         img = draw_traj_on_topview(img, overall_trajs[:, np.argwhere(agent.agent_id == agent_id[0]).item(), :2],
            #                                    self.obs_len, self.x_range, self.y_range, self.map_size, (0, 255, 255))
            #
            # # cv2.imshow("", img)
            # # cv2.waitKey(0)
            #
            # grid_size = 0.1
            # num_grid = 800
            # x = create_ROI(grid_size, num_grid).reshape(num_grid ** 2, 2)
            # for a in range(0, obs_traj.shape[1]):
            #
            #     a_id = agent_id[0, a]
            #     # a_token = scene.id_2_token_lookup[a_id]
            #     # agent = scene.agent_dict[a_token]
            #     if (a_id != 2):
            #         continue
            #
            #
            #     cur_pos = obs_traj[-1, a, :2].reshape(1, 2)
            #     # prev_pos = obs_traj[-2, a, :2].reshape(1, 2)
            #     cur_R = R_map[a]
            #
            #     x_rot = np.matmul(cur_R, x.T).T + cur_pos
            #     aa = pooling_operation(img, x_rot, self.x_range, self.y_range, self.map_size)
            #     b = aa.reshape(num_grid, num_grid, 3).astype('uint8')
            #     # b = cv2.resize(b, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            #
            #     obs_traj_ego = np.matmul(R_traj[a], (overall_trajs[:, a, :2]-cur_pos).T).T
            #     c = np.zeros(shape=(num_grid, num_grid, 3))
            #     c = draw_traj_on_topview(c, obs_traj_ego, self.obs_len, (-40, 40), (-40, 40), num_grid, (0, 255, 255))
            #
            #     d = np.hstack([b, c])
            #     cv2.imshow('test', d.astype('uint8'))
            #     cv2.waitKey(0)
            # debug -------------

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
            Ht, Hb = 0, 0

        return obs_traj, future_traj, obs_traj_e, future_traj_e, feature_topview, R_map, R_traj, Ht, Hb, num_agents, agent_id, scene

    def extract_data_from_scene(self, scene, isTrain=False):

        # trajectory
        num_total_agents = len(scene.agent_dict)
        agent_id = np.zeros(shape=(1, num_total_agents))
        category = np.zeros(shape=(1, num_total_agents)) # 0: vehicle, 1:pedestrian
        trajectories = np.full(shape=(self.obs_len + self.pred_len, num_total_agents, 3), fill_value=np.nan)
        trajectories_proc = np.full(shape=(self.obs_len + self.pred_len, num_total_agents, 3), fill_value=np.nan)
        for idx, track_id in enumerate(scene.agent_dict):
            agent_id[0, idx] = scene.agent_dict[track_id].agent_id
            if (scene.agent_dict[track_id].type == 'vehicle'):
                category[0, idx] = 0
            else:
                category[0, idx] = 1

            trajectory = scene.agent_dict[track_id].trajectory
            trajectories[:, idx, :] = trajectory[:, 1:]

            trajectory_proc = scene.agent_dict[track_id].trajectory_proc
            trajectories_proc[:, idx, :] = trajectory_proc[:, 1:]

        # find agents inside the limit range
        limit_range = self.limit_range
        if (np.random.rand(1) < self.limit_range_change_prob and isTrain):
            limit_range = float(random.randint(15, int(self.limit_range)))

        valid_flag = np.sqrt(np.sum(trajectories[self.obs_len - 1, :, :2] ** 2, axis=1)) < limit_range
        trajectories = trajectories[:, valid_flag, :]
        trajectories_proc = trajectories_proc[:, valid_flag, :]
        agent_id = agent_id[:, valid_flag]
        category = category[:, valid_flag]


        # find agents who has valid observation
        valid_flag = np.sum(trajectories[self.obs_len - self.min_obs_len:self.obs_len, :, 0], axis=0) > -1000
        trajectories = trajectories[:, valid_flag, :]
        trajectories_proc = trajectories_proc[:, valid_flag, :]
        agent_id = agent_id[:, valid_flag]
        category = category[:, valid_flag]


        # find trajectories that have full future traj,
        valid_flag = np.min(trajectories[self.obs_len:, :, 0], axis=0) > -1000
        trajectories = trajectories[:, valid_flag, :]
        trajectories_proc = trajectories_proc[:, valid_flag, :]
        agent_id = agent_id[:, valid_flag]
        category = category[:, valid_flag]

        # NOTE --------------------------------------------------
        # Augmentation: add random noise to observation
        if (isTrain and np.random.rand(1) < self.pos_noise_prob):
        # if (True):
            std = self.pos_noise_std
            seq_len, batch, dim = trajectories[:self.obs_len].shape
            noise = std * 2.0 * (np.random.rand(seq_len, batch, dim) - 0.5)
            # noise = 2.0 * np.ones_like(trajectories[:self.obs_len])
            noise[:, 0, :] = 0
            trajectories[:self.obs_len] += noise

        # NOTE --------------------------------------------------
        # topview img (Augmentation: lane color shift)
        feature_topview = self.map.make_topview_map_TypeD(scene.agent_dict['EGO'].pose, self.x_range, self.y_range,
                                                    self.map_size, trajectories[:self.obs_len], category, isTrain)

        # NOTE --------------------------------------------------
        # Augmentation: agent sampling
        if (isTrain):

            # calc speed
            disp = (trajectories[1:] - trajectories[:-1])[:, :, :2]
            speed_kmph_max = 3.6 * self.target_sample_period * np.max(np.sqrt(np.sum(disp ** 2, axis=2))[self.obs_len-1:], axis=0)

            # find agent with speed greater than 10 or smaller than 10
            high_speed = speed_kmph_max >= self.agent_sampling_thr
            high_speed[0] = True

            rand_bool = np.random.rand(trajectories.shape[1]) < self.agent_sampling_prob
            rand_bool[high_speed] = True

            valid_flag = np.logical_or(high_speed, rand_bool)
            if (np.count_nonzero(valid_flag) == 1):
                valid_flag = np.full_like(valid_flag, fill_value=True)

            # remove low speed vehicles
            trajectories = trajectories[:, valid_flag, :]
            trajectories_proc = trajectories_proc[:, valid_flag, :]
            agent_id = agent_id[:, valid_flag]
            category = category[:, valid_flag]

        # remove pedestrian
        valid_flag = category[0, :] == 0
        trajectories = trajectories[:, valid_flag, :]
        trajectories_proc = trajectories_proc[:, valid_flag, :]
        agent_id = agent_id[:, valid_flag]
        num_agents = trajectories.shape[1]

        # split into observation and future
        obs_traj = trajectories[:self.obs_len]
        future_traj = trajectories[self.obs_len:]
        future_traj_proc = trajectories_proc[self.obs_len:]

        # NOTE --------------------------------------------------
        # Augmentation: random flip
        isFlip = False
        if (np.random.rand(1) < self.random_flip_prob and isTrain):
            feature_topview = np.copy(np.fliplr(feature_topview))
            obs_traj[:, :, 1] = -1 * obs_traj[:, :, 1]
            future_traj[:, :, 1] = -1 * future_traj[:, :, 1]
            future_traj_proc[:, :, 1] = -1 * future_traj_proc[:, :, 1]
            isFlip = True


        # rotation matrix
        R_map, Ht, Hb = self.rotation_matrix(scene, agent_id, obs_traj, isFlip=isFlip, isTrain=isTrain)
        R_traj = np.linalg.inv(R_map)

        # 'nan' in ego-centric trajectories becomes zero
        nan_pos_obs = np.isnan(obs_traj[:, :, 2])
        nan_pos_future = np.isnan(future_traj[:, :, 2])

        obs_traj_e = []
        future_traj_e = []
        for a in range(num_agents):
            cur_R = R_traj[a]  # 2 x 2
            cur_center = obs_traj[-1, a, :2].reshape(1, 2)  # 1 x 2
            cur_obs_traj = obs_traj[:, a, :2]  # seq_len x 2
            cur_future_traj = future_traj[:, a, :2]
            cur_future_traj_proc = future_traj_proc[:, a, :2]

            cur_obs_traj_e = np.matmul(cur_R, (cur_obs_traj - cur_center).T).T  # seq_len x 2
            cur_future_traj_e = np.matmul(cur_R, (cur_future_traj - cur_center).T).T
            cur_future_traj_e_proc = np.matmul(cur_R, (cur_future_traj_proc - cur_center).T).T

            obs_traj_e.append(cur_obs_traj_e.reshape(self.obs_len, 1, 2))

            if (self.is_use_proc_traj == 0):
                future_traj_e.append(cur_future_traj_e.reshape(self.pred_len, 1, 2))
            else:
                if (np.count_nonzero(np.isnan(cur_future_traj_e_proc)) == 0):
                    future_traj_e.append(cur_future_traj_e_proc.reshape(self.pred_len, 1, 2))
                else:
                    future_traj_e.append(cur_future_traj_e.reshape(self.pred_len, 1, 2))


        obs_traj_e = np.concatenate(obs_traj_e, axis=1)
        future_traj_e = np.concatenate(future_traj_e, axis=1)

        obs_traj_e[nan_pos_obs] = 0
        future_traj_e[nan_pos_future] = 0

        return obs_traj, future_traj, obs_traj_e, future_traj_e, feature_topview, \
               R_map, R_traj, Ht, Hb, num_agents, agent_id

    def load_voxels(self, sample_token):
        file_name = sample_token + '.pkl'
        f = open(os.path.join(self.voxel_save_dir, file_name), 'rb')
        voxel_dict = pickle.load(f)
        f.close()

        assert (voxel_dict['x_min'] == self.x_range[0])
        assert (voxel_dict['y_min'] == self.y_range[0])
        assert (voxel_dict['z_min'] == self.z_range[0])
        assert (voxel_dict['x_max'] == self.x_range[1])
        assert (voxel_dict['y_max'] == self.y_range[1])
        assert (voxel_dict['z_max'] == self.z_range[1])
        assert (voxel_dict['num_lidar_sweeps'] == self.num_lidar_sweeps)

        return voxel_dict['voxels'].astype('float')

    def rotation_matrix(self, scene, agent_ids, obs_trajs, isFlip=False, isTrain=True):

        direction = -1.0 if isFlip else 1
        pos_diff = (obs_trajs[1:] - obs_trajs[:-1])[-1][:, :2]  # batch x 2
        speed_mps = self.target_sample_period * np.sqrt(np.sum(pos_diff ** 2, axis=1))  # batch

        Rs, Ht, Hb = [], [], []
        for _, aid in enumerate(agent_ids[0]):

            key = scene.id_2_token_lookup[aid]

            if (key == 'EGO'):
                R = np.eye(2)
                Ht.append(0)
                Hb.append(0)
            else:
                agent = scene.agent_dict[key]
                trajectory = obs_trajs[:, _, :]

                heading_t = self.heading_from_points(trajectory[-2, :2].reshape(1, 2),
                                                     trajectory[-1, :2].reshape(1, 2))

                heading_b = self.heading_from_points(agent.bbox_e[1, :].reshape(1, 2),
                                                     agent.bbox_e[0, :].reshape(1, 2))

                Ht.append(heading_t)
                Hb.append(heading_b)

                if (speed_mps[_] < 1):
                    heading_f = direction * heading_b
                else:
                    heading_f = heading_t

                if (self.is_bbox_heading):
                    heading_f = direction * heading_b

                # Add random noise to heading
                if (isTrain and self.add_heading_noise == 1):
                    rand_deg = self.heading_noise_deg_std * 2.0 * (np.random.rand(1) - 0.5) # -self.h_n_d_s ~ self.h_n_d_s
                    rand_rad = np.deg2rad(rand_deg)
                    heading_f += rand_rad

                    if (heading_f < -1*np.pi):
                        heading_f = -1*np.pi

                    if (heading_f > np.pi):
                        heading_f = np.pi

                m_cos = np.cos(heading_f)
                m_sin = np.sin(heading_f)
                R = np.array([m_cos, -1 * m_sin, m_sin, m_cos]).reshape(2, 2)

            Rs.append(np.expand_dims(R, axis=0))

        return np.concatenate(Rs, axis=0), np.array(Ht), np.array(Hb)

    def heading_from_points(self, p0, p1):
        '''
        p0, p1 (1 x 2)
        '''

        vec1 = p1 - p0
        x1 = vec1[0, 0]
        y1 = vec1[0, 1]
        heading = np.arctan2(y1, x1)

        return heading
