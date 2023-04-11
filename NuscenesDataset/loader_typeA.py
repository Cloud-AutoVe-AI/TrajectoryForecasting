from NuscenesDataset.visualization import *
from NuscenesDataset.map import Map
from NuscenesDataset.preprocess import DatasetBuilder
import NuscenesDataset.nuscenes.nuscenes as nuscenes_module
from torch.utils.data import Dataset
from utils.functions import read_config
from pyquaternion import Quaternion
from NuscenesDataset.scene import AgentCentricScene


class DatasetLoader(Dataset):

    def __init__(self, args, dtype, isTrain=True):

        # exp settings
        exp_type = 'train' if isTrain else 'test'

        # nuscenes api
        self.nusc = nuscenes_module.NuScenes(version='v1.0-trainval', dataroot=args.dataset_path, verbose=False)

        # nuscenes map api
        self.map = Map(args, self.nusc)

        # params
        config = read_config()

        self.dtype = dtype
        self.obs_len = int(args.past_horizon_seconds * args.target_sample_period)
        self.pred_len = int(args.future_horizon_seconds * args.target_sample_period)
        self.min_obs_len = int(args.min_past_horizon_seconds * args.target_sample_period)

        self.limit_range = args.limit_range
        self.x_range = (args.x_range_min, args.x_range_max)
        self.y_range = (args.y_range_min, args.y_range_max)
        self.z_range = (args.z_range_min, args.z_range_max)
        self.map_size = args.map_size
        self.lidar_map_ch_dim = args.lidar_map_ch_dim
        self.num_lidar_sweeps = args.num_lidar_sweeps

        self.use_preprocessed_lidar = args.use_preprocessed_lidar
        self.voxel_save_dir = config['Nuscenes']['voxel_dir']

        self.stop_agents_remove_prob = args.stop_agents_remove_prob
        self.limit_range_change_prob = args.limit_range_change_prob

        # debug ---
        self.h_noise_std = args.h_noise_std
        self.heading_est_method = args.heading_est_method
        self.add_noise_for_heading_est = args.add_noise_for_heading_est
        # debug ---

        # checks existance of dataset file and create
        save_path = config['Nuscenes']['preproc_dataset_path'] + '/%dsec_%dsec' % (args.past_horizon_seconds, args.future_horizon_seconds)
        if (os.path.exists(save_path)==False):
            os.mkdir(save_path)
        file_name = 'nuscenes_%s_cat%d.cpkl' % (exp_type, args.category_filtering_method) # update, 211005

        builder = DatasetBuilder(args, map=self.map, isTrain=isTrain)
        if (os.path.exists(os.path.join(save_path, file_name))==False):
            builder.make_preprocessed_data(os.path.join(save_path, file_name), exp_type)

        # load dataset file
        with open(os.path.join(save_path, file_name), 'rb') as f:
            dataset = dill.load(f, encoding='latin1')
            print(">> {%s} is loaded .." % (os.path.join(save_path, file_name)))  # update, 211005

        self.train_data = dataset[0]
        self.valid_data = dataset[1]
        self.test_data = dataset[2]

        if (self.use_preprocessed_lidar == 1):
            builder.preprocess_lidar_sweeps(self.train_data)
            builder.preprocess_lidar_sweeps(self.valid_data)
            builder.preprocess_lidar_sweeps(self.test_data)

        self.num_train_scenes = len(self.train_data)
        self.num_valid_scenes = len(self.valid_data)
        self.num_test_scenes = len(self.test_data)

        if (isTrain):
            self.analyze_dataset(self.train_data)
        else:
            self.analyze_dataset(self.test_data)

        print(">> Loader is loaded from {%s} " % os.path.basename(__file__))


    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):

        # current scene data
        scene = self.train_data[idx]

        obs_traj, future_traj, obs_traj_e, future_traj_e, feature_topview, R_map, R_traj, num_agents, agent_id, valid_loss_flag = \
            self.extract_data_from_scene(scene, isTrain=True)

        obs_traj = torch.from_numpy(obs_traj).type(self.dtype)
        future_traj = torch.from_numpy(future_traj).type(self.dtype)
        obs_traj_e = torch.from_numpy(obs_traj_e).type(self.dtype)
        future_traj_e = torch.from_numpy(future_traj_e).type(self.dtype)
        feature_topview = torch.from_numpy(feature_topview).permute(2, 0, 1).type(self.dtype)
        feature_topview = torch.unsqueeze(feature_topview, dim=0)
        R_map = torch.from_numpy(R_map).type(self.dtype)
        R_traj = torch.from_numpy(R_traj).type(self.dtype)

        return obs_traj, future_traj, obs_traj_e, future_traj_e, feature_topview, R_map, R_traj, num_agents, valid_loss_flag

    def next_sample(self, index, mode):

        if (mode == 'valid'):
            scene = self.valid_data[index]
        else:
            scene = self.test_data[index]

        obs_traj, future_traj, obs_traj_e, future_traj_e, feature_topview, R_map, R_traj, num_agents, agent_id, valid_loss_flag = \
            self.extract_data_from_scene(scene, isTrain=False)

        # # debug -------------
        # pc = np.sum(feature_topview[:, :, :20*self.num_lidar_sweeps], axis=2).reshape(self.map_size, self.map_size, 1)
        # lines = feature_topview[:, :, 20 * self.num_lidar_sweeps + 0].reshape(self.map_size, self.map_size, 1)
        # da = feature_topview[:, :, 20 * self.num_lidar_sweeps + 1].reshape(self.map_size, self.map_size, 1)
        # road = feature_topview[:, :, 20 * self.num_lidar_sweeps + 2].reshape(self.map_size, self.map_size, 1)
        # inter = feature_topview[:, :, 20 * self.num_lidar_sweeps + 3].reshape(self.map_size, self.map_size, 1)
        # pedc = feature_topview[:, :, 20 * self.num_lidar_sweeps + 4].reshape(self.map_size, self.map_size, 1)
        # walk = feature_topview[:, :, 20 * self.num_lidar_sweeps + 5].reshape(self.map_size, self.map_size, 1)
        # stop = feature_topview[:, :, 20 * self.num_lidar_sweeps + 6].reshape(self.map_size, self.map_size, 1)
        #
        #
        # pc[pc < 0] = 0
        # lines[lines < 0] = 0
        # da[da < 0] = 0
        # road[road < 0] = 0
        # inter[inter < 0] = 0
        # pedc[pedc < 0] = 0
        # walk[walk < 0] = 0
        # stop[stop < 0] = 0
        #
        #
        # ch0 = 255 * (lines + da + road + inter) / 4
        # ch1 = 255 * (pedc + walk + stop)
        # ch2 = 255 * pc
        #
        #
        # img = np.concatenate([ch0.astype('uint8'), ch1.astype('uint8'), ch2.astype('uint8')], axis=2)
        #
        # overall_trajs = np.concatenate([obs_traj[:, :, :2], future_traj[:, :, :2]], axis=0)
        # for _, id in enumerate(agent_id[0, :]):
        #     agent_key = scene.id_2_token_lookup[int(id)]
        #     agent = scene.agent_dict[agent_key]
        #
        #     if (agent.agent_id in agent_id[0, :].tolist()):
        #
        #         img = draw_bbox_on_topview(img, agent.trajectory[self.obs_len-1, 1:3].reshape(1, 2), R_map[_], agent.bbox(),
        #                                    self.x_range, self.y_range, self.map_size, (0, 0, 255))
        #
        #         img = draw_traj_on_topview(img, overall_trajs[:, np.argwhere(agent.agent_id == agent_id[0]).item(), :2],
        #                                    self.obs_len, self.x_range, self.y_range, self.map_size, (0, 255, 255))
        #
        # cv2.imshow("", img)
        # cv2.waitKey(0)
        #
        # for i in range(len(feature_topview)):
        #     if (i > 0):
        #         cv2.imshow("", feature_topview[i].astype('uint8'))
        #         cv2.waitKey(0)
        #
        # grid_size = 0.1
        # num_grid = 400
        # x = create_ROI(grid_size, num_grid).reshape(num_grid ** 2, 2)
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
        # # check ---

        # return obs_traj, future_traj, obs_traj_e, future_traj_e, feature_topview, R_map, R_traj, num_agents, agent_id, scene, valid_loss_flag
        return obs_traj, future_traj, obs_traj_e, future_traj_e, feature_topview, R_map, R_traj, num_agents, agent_id, scene, valid_loss_flag

    def extract_data_from_scene(self, scene, isTrain=True):

        scene_location = scene.city_name
        lidar_now_token = scene.lidar_token_seq[self.obs_len-1]
        lidar_now_data = self.nusc.get('sample_data', lidar_now_token)
        ego_pose = self.nusc.get('ego_pose', lidar_now_data['ego_pose_token'])


        # trajectory ---------
        num_total_agents = scene.num_agents
        agent_id = np.zeros(shape=(1, num_total_agents))
        trajectories = np.full(shape=(self.obs_len+self.pred_len, num_total_agents, 3), fill_value=np.nan)
        heading_traj = np.zeros(shape=(num_total_agents))
        heading_bbox = np.zeros(shape=(num_total_agents))
        speed = np.zeros(shape=(num_total_agents))
        # attribute = np.zeros(shape=(num_total_agents, 3)) # 0: stop, 1: moving, 2: turn,
        for idx, track_id in enumerate(scene.agent_dict):
            agent_id[0, idx] = scene.agent_dict[track_id].agent_id
            trajectory = scene.agent_dict[track_id].trajectory
            trajectories[:, idx, :] = trajectory[:, 1:]

            heading_traj[idx] = scene.agent_dict[track_id].heading_traj
            heading_bbox[idx] = scene.agent_dict[track_id].yaw
            speed[idx] = scene.agent_dict[track_id].speed

            # if ('stop' in scene.agent_dict[track_id].status):
            #     attribute[idx, 0] = 1
            #
            # if ('moving' in scene.agent_dict[track_id].status):
            #     attribute[idx, 1] = 1
            #
            # if ('turn' in scene.agent_dict[track_id].status):
            #     attribute[idx, 2] = 1

        # find agents inside the limit range
        limit_range = self.limit_range
        if (np.random.rand(1) < self.limit_range_change_prob and isTrain):
            limit_range = float(random.randint(15, int(self.limit_range)))

        valid_flag = np.sqrt(np.sum(trajectories[self.obs_len-1, :, :2] ** 2, axis=1)) < limit_range
        trajectories = trajectories[:, valid_flag, :]
        agent_id = agent_id[:, valid_flag]
        heading_traj = heading_traj[valid_flag]
        heading_bbox = heading_bbox[valid_flag]
        speed = speed[valid_flag]
        # attribute = attribute[valid_flag]


        # # remove stop agent
        # if (self.stop_agents_remove_prob > np.random.rand(1) and isTrain):
        #     valid_flag = attribute[:, 1] == 1 # consider only moving objects
        #     valid_flag[0] = True # ego should be included
        #     trajectories = trajectories[:, valid_flag, :]
        #     agent_id = agent_id[:, valid_flag]
        #     heading_traj = heading_traj[valid_flag]
        #     heading_bbox = heading_bbox[valid_flag]
        #     speed = speed[valid_flag]


        # # find agents who has valid observation
        # valid_flag = np.sum(trajectories[self.obs_len-self.min_obs_len:self.obs_len, :, 0], axis=0) > -1000
        # trajectories = trajectories[:, valid_flag, :]
        # agent_id = agent_id[:, valid_flag]
        # heading_traj = heading_traj[valid_flag]
        # heading_bbox = heading_bbox[valid_flag]
        # speed = speed[valid_flag]


        # preprocess stopped agents
        num_agents = trajectories.shape[1]
        # for a in range(num_agents):
        #     a_token = scene.id_2_token_lookup[agent_id[0, a]]
        #     agent = scene.agent_dict[a_token]
        #     if ('stop' in agent.status):
        #         trajectories[:, a, 0] = trajectories[self.obs_len - 1, a, 0] + 0.1 * np.random.rand(1)
        #         trajectories[:, a, 1] = trajectories[self.obs_len - 1, a, 1] + 0.1 * np.random.rand(1)
        #         trajectories[:, a, 2] = trajectories[self.obs_len - 1, a, 2] + 0.1 * np.random.rand(1)


        # find trajectories that have full future traj,
        valid_loss_flag = ~np.isnan(np.sum(trajectories[self.obs_len:, :, 0], axis=0))
        if (num_agents == 1):
            valid_loss_flag = valid_loss_flag.reshape(1, 1)

        # split into observation and future
        obs_traj = trajectories[:self.obs_len]
        future_traj = trajectories[self.obs_len:]

        # rotation matrix
        R_map = self.make_rot_matrix(heading_traj, heading_bbox, speed)
        R_traj = np.linalg.inv(R_map)


        # 'nan' in ego-centric trajectories becomes zero
        nan_pos_obs = np.isnan(obs_traj[:, :, 2])
        nan_pos_future = np.isnan(future_traj[:, :, 2])

        obs_traj_e = []
        future_traj_e = []
        for a in range(num_agents):
            cur_R = R_traj[a] # 2 x 2
            cur_center = obs_traj[-1, a, :2].reshape(1, 2) # 1 x 2
            cur_obs_traj = obs_traj[:, a, :2] # seq_len x 2
            cur_future_traj = future_traj[:, a, :2]

            cur_obs_traj_e = np.matmul(cur_R, (cur_obs_traj - cur_center).T).T # seq_len x 2
            cur_future_traj_e = np.matmul(cur_R, (cur_future_traj - cur_center).T).T

            obs_traj_e.append(cur_obs_traj_e.reshape(self.obs_len, 1, 2))
            future_traj_e.append(cur_future_traj_e.reshape(self.pred_len, 1, 2))

        obs_traj_e = np.concatenate(obs_traj_e, axis=1)
        future_traj_e = np.concatenate(future_traj_e, axis=1)

        obs_traj_e[nan_pos_obs] = 0
        future_traj_e[nan_pos_future] = 0


        # PC voxelization and topview
        if (self.use_preprocessed_lidar == 1):
            voxels_topview = self.load_voxels(scene.sample_token)
        else:
            lidar_past_sample_data = self.traverse_linked_list(lidar_now_data, 'sample_data', 'prev') + [lidar_now_data]
            past_lidar_sweeps_tokens = [_['token'] for _ in lidar_past_sample_data]
            voxels_topview = self.map.make_topview_voxels(past_lidar_sweeps_tokens[-self.num_lidar_sweeps:], self.x_range, self.y_range, self.z_range, self.map_size)

        # HDmap topview
        map_topview = self.map.make_topview_map(ego_pose, scene_location, self.x_range, self.y_range, self.map_size)
        feature_topview = np.concatenate([voxels_topview, map_topview], axis=2)

        # assert (np.count_nonzero(np.isnan(obs_traj_e)) == 0)
        # assert (np.count_nonzero(np.isnan(future_traj_e)) == 0)
        # assert (np.count_nonzero(np.isnan(feature_topview)) == 0)
        # assert (np.count_nonzero(np.isnan(R_map)) == 0)
        # assert (np.count_nonzero(np.isnan(R_traj)) == 0)

        return obs_traj, future_traj, obs_traj_e, future_traj_e, feature_topview, \
               R_map, R_traj, num_agents, agent_id, valid_loss_flag.reshape(1, num_agents)

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

    def make_rot_matrix(self, heading_traj, heading_bbox, speed):

        num_agents = len(heading_traj)
        heading_traj_copy = np.copy(heading_traj)

        # select heading estimation method (0: traj, 1: bbox)
        if (self.heading_est_method == 0):
            heading_traj[speed < 3] = heading_bbox[speed < 3]
        else:
            heading_traj[1:] = heading_bbox[1:]

        nan_heading_idx = np.isnan(heading_bbox)
        heading_traj[nan_heading_idx] = heading_traj_copy[nan_heading_idx]

        # calc rot matrix
        m_cos = np.cos(heading_traj).reshape(num_agents, 1)
        m_sin = np.sin(heading_traj).reshape(num_agents, 1)
        R = np.concatenate([m_cos, -1 * m_sin, m_sin, m_cos], axis=1).reshape(num_agents, 2, 2)

        return R

    def traverse_linked_list(self, obj, tablekey, direction, inclusive=False):
        return nuscenes_module.traverse_linked_list(self.nusc, obj, tablekey, direction, inclusive)

    def analyze_dataset(self, data):

        num_scenes = len(data)
        num_total_agents = 0

        num_vehicles = 0
        num_pedestrians = 0
        num_moving_vehicles = 0
        num_turning_vehicles = 0
        num_stopped_vehicles = 0

        for _, scene in enumerate(data):

            num_total_agents += scene.num_agents

            for idx, track_id in enumerate(scene.agent_dict):
                type = scene.agent_dict[track_id].type
                status = scene.agent_dict[track_id].status

                if ('vehicle' in type):
                    num_vehicles += 1

                if ('pedestrian' in type):
                    num_pedestrians += 1

                if ('vehicle' in type and 'moving' in status):
                    num_moving_vehicles += 1

                if ('vehicle' in type and 'turn' in status):
                    num_turning_vehicles += 1

                if ('vehicle' in type and 'stop' in status):
                    num_stopped_vehicles += 1

        print(">> num scenes %d, num total agents %d" % (num_scenes, num_total_agents))
        print(">> num vehicles %d (moving %d, stopped %d, turning %d)" % (num_vehicles, num_moving_vehicles, num_stopped_vehicles, num_turning_vehicles))

    # update, 211125
    def refactoring(self, scene, isTrain=True):

        samples = []

        # current sample (current time)
        num_total_agents = scene.num_agents
        sample = self.nusc.get('sample', scene.sample_token)
        lidar_sample_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        ref_ego_pose = self.nusc.get('ego_pose', lidar_sample_data['ego_pose_token'])
        R_e2g = Quaternion(ref_ego_pose['rotation']).rotation_matrix
        R_g2e = np.linalg.inv(R_e2g)
        trans_g_e = np.array(ref_ego_pose['translation']).reshape(1, 3)


        # all agent's trajectories
        agent_ids = np.zeros(shape=(1, num_total_agents))
        trajectories = np.full(shape=(self.obs_len+self.pred_len, num_total_agents, 3), fill_value=np.nan)
        bboxes = np.full(shape=(num_total_agents, 8, 3), fill_value=np.nan)
        for idx, track_id in enumerate(scene.agent_dict):
            agent_ids[0, idx] = scene.agent_dict[track_id].agent_id
            trajectory = scene.agent_dict[track_id].trajectory
            trajectories[:, idx, :] = trajectory[:, 1:]
            bboxes[idx, :, :] = scene.agent_dict[track_id].bbox_3d().T


        # find agents inside the limit range
        valid_flag = np.sqrt(np.sum(trajectories[self.obs_len-1, :, :2] ** 2, axis=1)) < self.limit_range
        trajectories = trajectories[:, valid_flag, :]
        agent_ids = agent_ids[:, valid_flag]
        bboxes = bboxes[valid_flag, :, :]

        # find agents who has valid observation
        valid_flag = np.sum(trajectories[self.obs_len - self.min_obs_len:self.obs_len, :, 0], axis=0) > -1000
        trajectories = trajectories[:, valid_flag, :]
        agent_ids = agent_ids[:, valid_flag]
        bboxes = bboxes[valid_flag, :, :]

        # transform to global coordiante system
        trajectories_g = np.copy(trajectories) # seq_len x batch x 3
        for a in range(agent_ids.shape[1]):
            trajectories_g[:, a, :] = self.map._transform_pc_inv(R_e2g, trans_g_e, trajectories[:, a, :])


        # debug ----------
        # trajectories_e = self.map._transform_pc(R_g2e, trans_g_e, trajectories_g[:, 0, :])
        # bboxes_e = self.map._transform_pc(R_g2e, trans_g_e, bboxes_g[0, :, :])
        #
        # map = self.map.make_topview_map_agent_centric(trans_g_e[0], R_e2g, R_g2e, scene.city_name, self.x_range,
        #                                                   self.y_range, self.map_size).astype('uint8')
        # img = np.sum(map, axis=2).reshape(self.map_size, self.map_size, 1).repeat(3, axis=2).astype('float')
        # img = 128 * img / np.max(img)
        #
        # img = _draw_centerlines(img.astype('uint8'), trajectories_e, self.x_range, self.y_range, self.map_size)
        # img = _draw_bbox_on_topview(img.astype('uint8'), bboxes_e, (0, 0, 255), self.x_range, self.y_range,
        #                            self.map_size)
        # cv2.imshow("", img.astype('uint8'))
        # cv2.waitKey(0)
        # debug -----------

        # for all agents
        for a in range(agent_ids.shape[1]):

            agent_track_id = scene.id_2_token_lookup[agent_ids[0, a]]
            agent = scene.agent_dict[agent_track_id]
            if (agent_track_id == 'EGO'):
                R_g2a = R_g2e
                R_a2g = R_e2g
            else:
                ann = self.nusc.get('sample_annotation', agent_track_id)
                R_a2g = Quaternion(ann['rotation']).rotation_matrix
                R_g2a = np.linalg.inv(R_a2g)

            trans_a = trajectories_g[self.obs_len - 1, a, :].reshape(1, 3)

            # debug ----------
            # map = self.map.make_topview_map_agent_centric(trans_a[0], R_a2g, R_g2a, scene.city_name, self.x_range,
            #                                               self.y_range, self.map_size).astype('uint8')
            # img = np.sum(map, axis=2).reshape(self.map_size, self.map_size, 1).repeat(3, axis=2).astype('float')
            # img = 128 * img / np.max(img)
            # debug -----------


            # skip if the target agent doesn't has full trajectory
            FLAG = np.min(trajectories_g[self.obs_len:, a, 0]) > -1000
            if not FLAG:
                continue

            # bboxes
            bboxes_g = np.copy(bboxes)
            bboxes_g[a, :, :] = self.map._transform_pc_inv(R_a2g, trans_a, bboxes[a, :, :])

            # trajectories
            trajectories_a = np.copy(trajectories_g)
            for aa in range(agent_ids.shape[1]):
                trajectories_a[:, aa, :] = self.map._transform_pc(R_g2a, trans_a, trajectories_g[:, aa, :])

                # debug ----
                # img = _draw_centerlines(img.astype('uint8'), trajectories_a[:, aa, :], self.x_range, self.y_range, self.map_size)
                #
                # if (a == aa):
                #     img = _draw_bbox_on_topview(img.astype('uint8'), bboxes[aa, :, :], (0, 0, 255), self.x_range, self.y_range, self.map_size)
                # debug ----

            agent_sample = AgentCentricScene(sample_token=scene.sample_token, agent_token=agent_track_id, city_name=scene.city_name)
            agent_sample.target_agent_index = agent_ids[0, a]
            agent_sample.trajectories = trajectories_a
            agent_sample.bboxes = bboxes_g
            agent_sample.R_a2g = R_a2g
            agent_sample.R_g2a = R_g2a
            agent_sample.trans_g = trans_a
            agent_sample.agent_ids = agent_ids
            agent_sample.possible_lanes = agent.possible_lanes
            samples.append(agent_sample)


            # debug ----
            # cv2.imshow("", img.astype('uint8'))
            # cv2.waitKey(0)
            # debug ----


        return samples

    # update, 211125
    def convert_to_agentcentric(self, _obs_traj, _future_traj, _pred_traj, agent_ids, agent_samples):

        '''
        Inputs are ego-centric (ego: AV)
        '''

        best_k = _pred_traj.shape[0]
        num_agents = len(agent_ids)

        # extend dims
        z_axis = np.expand_dims(_future_traj[:, :, 2].reshape(self.pred_len, num_agents, 1), axis=0)
        z_axis = np.repeat(z_axis, best_k, axis=0)
        _pred_traj = np.concatenate([_pred_traj, z_axis], axis=3)

        # ego-vehicle R & T
        idx = np.argwhere(agent_ids == 0)[0][0]
        assert (idx == 0)
        R_g2e = agent_samples[idx].R_g2a
        R_e2g = agent_samples[idx].R_a2g
        trans_g_e = agent_samples[idx].trans_g


        obs_traj, future_traj, pred_traj_k = [], [], []
        for i in range(num_agents):

            # ego-vehicle
            if (agent_ids[i] == 0):
                obs = _obs_traj[:, i, :].reshape(self.obs_len, 1, 3)
                future = _future_traj[:, i, :].reshape(self.pred_len, 1, 3)
                preds = _pred_traj[:, :, i, :].reshape(best_k, self.pred_len, 1, 3)

                obs_traj.append(obs)
                future_traj.append(future)
                pred_traj_k.append(preds)

            else:
                agent_id = agent_ids[i]
                idx = 0
                for a in range(num_agents):
                    if (agent_id == agent_samples[a].target_agent_index):
                        idx = a
                        break

                R_a2g = agent_samples[idx].R_a2g
                R_g2a = agent_samples[idx].R_g2a
                trans_g_a = agent_samples[idx].trans_g

                obs = _obs_traj[:, i, :]
                future = _future_traj[:, i, :]
                preds = _pred_traj[:, :, i, :]

                obs = self.map._transform_pc(R_g2a, trans_g_a, self.map._transform_pc_inv(R_e2g, trans_g_e, obs))
                future = self.map._transform_pc(R_g2a, trans_g_a, self.map._transform_pc_inv(R_e2g, trans_g_e, future))

                preds_k = []
                for k in range(best_k):
                    pred = self.map._transform_pc(R_g2a, trans_g_a,
                                                 self.map._transform_pc_inv(R_e2g, trans_g_e, preds[k, :, :]))
                    preds_k.append(np.expand_dims(pred, axis=0))
                preds = np.concatenate(preds_k, axis=0)

                obs_traj.append(np.expand_dims(obs, axis=1))
                future_traj.append(np.expand_dims(future, axis=1))
                pred_traj_k.append(np.expand_dims(preds, axis=2))

        obs_traj = np.concatenate(obs_traj, axis=1)
        future_traj = np.concatenate(future_traj, axis=1)
        pred_traj_k = np.concatenate(pred_traj_k, axis=2)

        return obs_traj, future_traj, pred_traj_k
