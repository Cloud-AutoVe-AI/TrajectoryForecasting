from NuscenesDataset.visualization import *
from NuscenesDataset.map import Map
from NuscenesDataset.preprocess import DatasetBuilder
import NuscenesDataset.nuscenes.nuscenes as nuscenes_module
from torch.utils.data import Dataset
from utils.functions import read_config
from NuscenesDataset.scene import AgentCentricScene
from pyquaternion import Quaternion

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

        # update, 220204
        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma = args.gamma

        self.dtype = dtype
        self.target_sample_period = args.target_sample_period
        self.obs_len = int(args.past_horizon_seconds * args.target_sample_period)
        self.pred_len = int(args.future_horizon_seconds * args.target_sample_period)
        self.min_obs_len = int(args.min_past_horizon_seconds * args.target_sample_period)

        self.limit_range = args.limit_range
        self.x_range = (args.x_range_min, args.x_range_max)
        self.y_range = (args.y_range_min, args.y_range_max)
        self.z_range = (args.z_range_min, args.z_range_max)
        self.map_size = args.map_size

        self.is_crop_topview = args.is_crop_topview
        self.x_crop_range = (args.x_crop_min, args.x_crop_max)
        self.y_crop_range = (args.y_crop_min, args.y_crop_max)

        self.lidar_map_ch_dim = args.lidar_map_ch_dim
        self.num_lidar_sweeps = args.num_lidar_sweeps
        self.use_preprocessed_lidar = args.use_preprocessed_lidar
        self.voxel_save_dir = config['Nuscenes']['voxel_dir']

        self.best_k = args.best_k
        self.neg_traj_ratio = args.neg_traj_ratio
        self.collision_dist = args.collision_dist

        self.traj_set_eps = args.traj_set_eps
        file_name = './utils/trajectory_set_sp%d_ph%d_ep%d.pkl' % (self.target_sample_period, args.future_horizon_seconds, self.traj_set_eps)
        with open(file_name, 'rb') as f:
            self.trajectory_set = pickle.load(f)
            print(">> {%s} is loaded .." % file_name)


        # checks existance of dataset file and create
        save_path = config['Nuscenes']['preproc_dataset_path'] + '/%dsec_%dsec' % (args.past_horizon_seconds, args.future_horizon_seconds)
        if (os.path.exists(save_path)==False):
            os.mkdir(save_path)
        file_name = 'nuscenes_%s_cat%d.cpkl' % (exp_type, args.category_filtering_method)

        builder = DatasetBuilder(args, map=self.map, isTrain=isTrain)
        if (os.path.exists(os.path.join(save_path, file_name))==False):
            builder.make_preprocessed_data(os.path.join(save_path, file_name), exp_type)

        # load dataset file
        with open(os.path.join(save_path, file_name), 'rb') as f:
            dataset = dill.load(f, encoding='latin1')
            print(">> {%s} is loaded .." % (os.path.join(save_path, file_name)))

        self.train_data, self.valid_data = [], []
        for _, scene in enumerate(tqdm(dataset[0], desc='refactoring train data')):
            self.train_data += self.refactoring(scene)
        for _, scene in enumerate(tqdm(dataset[1], desc='refactoring valid data')):
            self.valid_data += self.refactoring(scene, isTrain=False)
        self.test_data = dataset[2]

        self.num_train_scenes = len(self.train_data)
        self.num_valid_scenes = len(self.valid_data)
        self.num_test_scenes = len(self.test_data)

        print(">> Loader is loaded from {%s} " % os.path.basename(__file__))
        print(">> num train/valid samples : %d / %d" % (self.num_train_scenes, self.num_valid_scenes))
        print(">> num test scenes : %d " % self.num_test_scenes)


    def __len__(self):
        return len(self.train_data)
        # return 32


    def __getitem__(self, idx):

        # current scene data
        obs_traj_ta, future_traj_ta, obs_traj_ngh, future_traj_ngh, map, neg_traj, dist2gt_cost, \
        dist2ngh_cost, drivable_cost, num_neighbors, valid_neighbor = self.extract_data_from_scene(self.train_data[idx])

        obs_traj_ta = torch.from_numpy(obs_traj_ta).type(self.dtype)
        future_traj_ta = torch.from_numpy(future_traj_ta).type(self.dtype)
        obs_traj_ngh = torch.from_numpy(obs_traj_ngh).type(self.dtype)
        future_traj_ngh = torch.from_numpy(future_traj_ngh).type(self.dtype)
        map = torch.from_numpy(map).permute(2, 0, 1).type(self.dtype)
        map = torch.unsqueeze(map, dim=0)

        neg_traj = torch.from_numpy(neg_traj).type(self.dtype) # best_k x seq_len x dim
        dist2gt_cost = torch.from_numpy(dist2gt_cost).type(self.dtype) # best_k x seq_len
        dist2ngh_cost = torch.from_numpy(dist2ngh_cost).type(self.dtype) # best_k x seq_len
        drivable_cost = torch.from_numpy(drivable_cost).type(self.dtype)  # best_k x seq_len

        neg_traj = torch.unsqueeze(neg_traj, dim=2) # best_k x seq_len x batch x dim
        dist2gt_cost = torch.unsqueeze(dist2gt_cost, dim=0) # 1 x best_k x seq_len
        dist2ngh_cost = torch.unsqueeze(dist2ngh_cost, dim=0)  # 1 x best_k x seq_len
        drivable_cost = torch.unsqueeze(drivable_cost, dim=0)  # 1 x best_k x seq_len

        return obs_traj_ta, future_traj_ta, obs_traj_ngh, future_traj_ngh, map, \
               neg_traj, dist2gt_cost, dist2ngh_cost, drivable_cost, num_neighbors, valid_neighbor

    def next_sample(self, index, mode):

        # current scene data
        if (mode == 'valid'):
            obs_traj_ta, future_traj_ta, obs_traj_ngh, future_traj_ngh, map, neg_traj, dist2gt_cost, dist2ngh_cost, \
            drivable_cost, num_neighbors, valid_neighbor = self.extract_data_from_scene(self.valid_data[index])

            return obs_traj_ta, future_traj_ta, obs_traj_ngh, future_traj_ngh, map, \
                   neg_traj, dist2gt_cost, dist2ngh_cost, drivable_cost, num_neighbors, valid_neighbor
        else:
            scene = self.test_data[index]
            agent_samples = self.refactoring(scene)

            obs_traj, future_traj, obs_traj_ngh, future_traj_ngh, map, num_neighbors, valid_neighbor, agent_ids, \
            neg_traj, dist2gt_cost, dist2ngh_cost, drivable_cost = [], [], [], [], [], [], [], [], [], [], [], []

            for i in range(len(agent_samples)):

                # agent_index
                agent_id = agent_samples[i].target_agent_index

                # current sample
                _obs_traj, _future_traj, _obs_traj_ngh, _future_traj_ngh, _map, _neg_traj, _dist2gt_cost, \
                _dist2ngh_cost, _drivable_cost, _num_neighbors, _valid_neighbor = self.extract_data_from_scene(agent_samples[i])

                _obs_traj = torch.from_numpy(_obs_traj).type(self.dtype)
                _future_traj = torch.from_numpy(_future_traj).type(self.dtype)
                _obs_traj_ngh = torch.from_numpy(_obs_traj_ngh).type(self.dtype)
                _future_traj_ngh = torch.from_numpy(_future_traj_ngh).type(self.dtype)
                _map = torch.from_numpy(_map).permute(2, 0, 1).type(self.dtype)
                _map = torch.unsqueeze(_map, dim=0)

                obs_traj.append(_obs_traj)
                future_traj.append(_future_traj)
                obs_traj_ngh.append(_obs_traj_ngh)
                future_traj_ngh.append(_future_traj_ngh)
                map.append(_map)
                num_neighbors.append(_num_neighbors)
                valid_neighbor.append(_valid_neighbor)
                agent_ids.append(agent_id)


            _len = [objs for objs in num_neighbors]
            cum_start_idx = [0] + np.cumsum(_len).tolist()
            seq_start_end = [[start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])]

            obs_traj = torch.cat(obs_traj, dim=1)
            future_traj = torch.cat(future_traj, dim=1)
            obs_traj_ngh = torch.cat(obs_traj_ngh, dim=1)
            future_traj_ngh = torch.cat(future_traj_ngh, dim=1)
            map = torch.cat(map, dim=0)
            seq_start_end = torch.LongTensor(seq_start_end)
            valid_neighbor = np.array(valid_neighbor)
            agent_ids = np.array(agent_ids)

            return [obs_traj, future_traj, obs_traj_ngh, future_traj_ngh, map, neg_traj, dist2gt_cost,
                    dist2ngh_cost, drivable_cost, seq_start_end, valid_neighbor, agent_ids, agent_samples, scene]


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
        yaw_g = np.full(shape=(self.obs_len + self.pred_len, num_total_agents, 1), fill_value=np.nan)
        category = np.ones(shape=(num_total_agents))  # 0: vehicle, 1:pedestrian
        for idx, track_id in enumerate(scene.agent_dict):
            agent_ids[0, idx] = scene.agent_dict[track_id].agent_id
            trajectory = scene.agent_dict[track_id].trajectory
            trajectories[:, idx, :] = trajectory[:, 1:]
            yaw_g[:, idx, :] = scene.agent_dict[track_id].yaw_global

            if ('vehicle' in scene.agent_dict[track_id].type or 'VEHICLE' in scene.agent_dict[track_id].type):
                category[idx] = 0

        # find agents inside the limit range
        valid_flag = np.sqrt(np.sum(trajectories[self.obs_len-1, :, :2] ** 2, axis=1)) < self.limit_range
        trajectories = trajectories[:, valid_flag, :]
        yaw_g = yaw_g[:, valid_flag, :]
        agent_ids = agent_ids[:, valid_flag]
        category = category[valid_flag]

        # find agents who has valid observation
        valid_flag = np.sum(trajectories[self.obs_len - self.min_obs_len:self.obs_len, :, 0], axis=0) > -1000
        trajectories = trajectories[:, valid_flag, :]
        yaw_g = yaw_g[:, valid_flag, :]
        agent_ids = agent_ids[:, valid_flag]
        category = category[valid_flag]

        # transform to global coordiante system
        trajectories_g = np.copy(trajectories) # seq_len x batch x 3
        for a in range(agent_ids.shape[1]):
            trajectories_g[:, a, :] = self.map._transform_pc_inv(R_e2g, trans_g_e, trajectories[:, a, :])


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

            # skip if pedestrian, (0: vehicle, 1:pedestrian)
            if (category[a] == 1):
                continue


            # skip if the target agent doesn't has full trajectory
            if not (np.min(trajectories_g[self.obs_len:, a, 0]) > -1000):
                continue

            # trajectories
            trajectories_a = np.copy(trajectories_g)
            for aa in range(agent_ids.shape[1]):
                trajectories_a[:, aa, :] = self.map._transform_pc(R_g2a, trans_a, trajectories_g[:, aa, :])

            agent_sample = AgentCentricScene(sample_token=scene.sample_token, agent_token=agent_track_id, city_name=scene.city_name)
            agent_sample.target_agent_index = agent_ids[0, a]
            agent_sample.trajectories = trajectories_a
            agent_sample.trajectories_g = trajectories_g
            agent_sample.yaw_g = yaw_g
            agent_sample.R_a2g = R_a2g
            agent_sample.R_g2a = R_g2a
            agent_sample.trans_g = trans_a
            agent_sample.agent_ids = agent_ids
            agent_sample.possible_lanes = agent.possible_lanes
            agent_sample.category = category
            samples.append(agent_sample)

        return samples

    def extract_data_from_scene(self, scene, isTrain=True):

        agent_ids = scene.agent_ids
        target_agent_index = scene.target_agent_index
        trajectories = scene.trajectories
        category = scene.category

        R_g2a = scene.R_g2a
        R_a2g = scene.R_a2g
        trans_g = scene.trans_g

        # find agents inside the limit range
        valid_flag = np.sqrt(np.sum(trajectories[self.obs_len-1, :, :2] ** 2, axis=1)) < self.limit_range
        trajectories = trajectories[:, valid_flag, :]
        agent_ids = agent_ids[:, valid_flag]
        category = category[valid_flag]

        # split into target agent and neighbors
        idx = np.argwhere(agent_ids[0, :] == target_agent_index)[0][0]

        num_agents = agent_ids.shape[1]
        trajectory_ta = np.full(shape=(self.obs_len+self.pred_len, 1, 3), fill_value=np.nan)
        trajectories_ngh, categories_ngh = [], []
        for l in range(num_agents):
            if (l == idx):
                trajectory_ta[:, :, :] = np.copy(trajectories[:, l, :].reshape(self.obs_len+self.pred_len, 1, 3))
            else:
                trajectories_ngh.append(np.copy(trajectories[:, l, :].reshape(self.obs_len+self.pred_len, 1, 3)))
                categories_ngh.append(category[l])

        num_neighbors = len(trajectories_ngh)
        if (num_neighbors == 0):
            trajectories_ngh = np.full(shape=(self.obs_len+self.pred_len, 1, 3), fill_value=np.nan)
            categories_ngh = np.zeros(shape=(1))
            valid_neighbor = False
            num_neighbors += 1
        else:
            trajectories_ngh = np.concatenate(trajectories_ngh, axis=1)
            categories_ngh = np.array(categories_ngh)
            valid_neighbor = True

        # split into observation and future
        obs_traj_ta = np.copy(trajectory_ta[:self.obs_len])
        future_traj_ta = np.copy(trajectory_ta[self.obs_len:])
        obs_traj_ngh = np.copy(trajectories_ngh[:self.obs_len])
        future_traj_ngh = np.copy(trajectories_ngh[self.obs_len:])

        # remove 'nan' in observation
        obs_traj_ta = self.remove_nan(obs_traj_ta)
        obs_traj_ngh = self.remove_nan(obs_traj_ngh)

        # nan to infinity
        chk_nan = np.isnan(future_traj_ngh)
        future_traj_ngh[chk_nan] = 1000.0

        # map
        map = self.map.make_topview_map_loadertypeF(trans_g[0], R_a2g, R_g2a, scene.city_name, self.x_range, self.y_range,
                                                    self.map_size, obs_traj_ngh, categories_ngh)

        if (self.is_crop_topview):
            map = self.crop_topview(map)

        # debug -----
        # img = (255.0 * map[:, :, :3]).astype('uint8')
        # img = _draw_agent_trajectories(img, trajectories, self.x_range, self.y_range, self.map_size, category)
        # cv2.imshow("", img.astype('uint8'))
        # cv2.waitKey(0)
        #
        # cv2.imshow("", (255*img_drivable).astype('uint8'))
        # cv2.waitKey(0)
        # debug -----

        # drivable space, update 220204
        if (self.beta > 0):
            R = self.map.transfrom_matrix(R_a2g, trans_g[0], inverse=False)
            v = np.dot(R[:3, :3], np.array([1, 0, 0]))
            yaw = np.arctan2(v[1], v[0])
            drivable_map = self.map.draw_drivable_space(trans_g[0], yaw, self.x_range, self.y_range, self.map_size, scene.city_name)
        else:
            drivable_map = np.zeros(shape=(self.map_size, self.map_size, 1))


        # --------------------------------------------------------
        # create label
        # --------------------------------------------------------
        target_traj = future_traj_ta[:, 0, :2] # seq_len x 2
        target_traj_repeat = np.repeat(np.expand_dims(target_traj, axis=0), self.trajectory_set.shape[0], axis=0) # num_trajs x seq_len x 2

        # TODO : how to select negative samples
        err = target_traj_repeat - self.trajectory_set # num_trajs x seq_len x 2
        dists = np.sum(np.sqrt(np.sum(err ** 2, axis=2)), axis=1) # num_trajs
        sorts = np.argsort(dists)[::-1][:int(self.neg_traj_ratio * self.trajectory_set.shape[0])].tolist()  # idx of decreasing distance
        random.shuffle(sorts)

        # TODO : performance variation with different best_k
        neg_idx = np.array(sorts[:self.best_k])
        neg_traj = self.trajectory_set[neg_idx] # best_k x seq_len x 2

        target_traj_repeat = np.repeat(np.expand_dims(target_traj, axis=0), self.best_k, axis=0) # best_k x seq_len x 2
        err = target_traj_repeat - neg_traj  # best_k x seq_len x 2
        dist2gt_cost = np.sqrt(np.sum(err ** 2, axis=2))  # best_k x seq_len

        drivable_cost = []
        for k in range(self.best_k):
            cur_neg_traj = neg_traj[k]
            seq_values = _pooling_operation(drivable_map, cur_neg_traj, self.x_range, self.y_range, self.map_size)
            drivable_cost.append(seq_values)

            # debug -----
            # print(seq_values)
            # input_map = 255*np.concatenate([drivable_map, drivable_map, drivable_map], axis=2)
            # img = _draw_agent_trajectories(input_map, np.expand_dims(neg_traj[k], axis=1), self.x_range, self.y_range, self.map_size, [0])
            # cv2.imshow("", img.astype('uint8'))
            # cv2.waitKey(0)
            # debug -----
        drivable_cost = np.concatenate(drivable_cost, axis=0)

        if (num_neighbors > 0):
            dist2ngh_cost = []
            for k in range(self.best_k):
                cur_neg_traj = neg_traj[k] # seq_len x 2
                cur_neg_traj_repeat = np.repeat(np.expand_dims(cur_neg_traj, axis=1), num_neighbors, axis=1) # seq_len x num_neighbors x 2

                err = cur_neg_traj_repeat - future_traj_ngh[:, :, :2] # seq_len x num_neighbors x 2
                min_dists = np.min(np.sqrt(np.sum(err ** 2, axis=2)), axis=1).reshape(1, self.pred_len)
                dist2ngh_cost.append(min_dists)
            dist2ngh_cost = np.concatenate(dist2ngh_cost, axis=0) # best_k x seq_len

            chk_high = dist2ngh_cost < self.collision_dist # close distance, high cost
            dist2ngh_cost[chk_high] = 1
            dist2ngh_cost[~chk_high] = 0

        else:
            dist2ngh_cost = np.zeros(shape=(self.best_k, self.pred_len))


        # --------------------------------------------------------
        # --------------------------------------------------------

        return obs_traj_ta, future_traj_ta, obs_traj_ngh, future_traj_ngh, map, \
               neg_traj, dist2gt_cost, dist2ngh_cost, drivable_cost, num_neighbors, valid_neighbor

    def calc_speed_heading(self, trajectory):

        '''
        trajectory : seq_len x batch x 3
        '''

        # # params
        seq_len, batch, dim = trajectory.shape

        # # speed (m/s) and heading (rad)
        traj = np.copy(np.squeeze(trajectory)[:, :2])
        pos_diff = np.zeros_like(traj)
        pos_diff[1:, :] = traj[1:] - traj[:-1]

        # speed
        speed_mps = self.target_sample_period * np.sqrt(np.sum(pos_diff ** 2, axis=1)).reshape(seq_len, 1, 1)

        # TODO : use heading from 3D-bbox
        # heading
        heading_rad = np.arctan2(pos_diff[:, 1], pos_diff[:, 0]).reshape(seq_len, 1, 1)
        # for i in range(1, seq_len):
        #     if (speed_mps[i] < 1e-2):
        #         heading_rad[i] = heading_rad[i-1]

        return np.concatenate([speed_mps, heading_rad, trajectory], axis=2)

    def remove_nan(self, seq):

        '''
        seq : seq_len x batch x 2
        '''

        seq_copy = np.copy(seq)
        for i in range(seq.shape[1]):
            cur_seq = np.copy(seq[:, i, :])
            if (np.count_nonzero(np.isnan(cur_seq[:-self.min_obs_len])) > 0):
                seq_copy[:-self.min_obs_len, i, :] = 0.0

        return seq_copy

    def convert_to_egocentric(self, _obs_traj, _future_traj, _pred_traj, agent_ids, agent_samples):

        best_k = _pred_traj.shape[0]
        num_agents = len(agent_samples)

        # extend dims
        z_axis = np.expand_dims(_future_traj[:, :, 2].reshape(self.pred_len, num_agents, 1), axis=0)
        z_axis = np.repeat(z_axis, best_k, axis=0)
        _pred_traj = np.concatenate([_pred_traj, z_axis], axis=3)

        # ego-vehicle R & T
        idx = np.argwhere(agent_ids == 0)[0][0]
        assert (idx == 0)
        R_g2e = agent_samples[idx].R_g2a
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
                R_a2g = agent_samples[i].R_a2g
                trans_g_a = agent_samples[i].trans_g

                obs = _obs_traj[:, i, :]
                future = _future_traj[:, i, :]
                preds = _pred_traj[:, :, i, :]

                obs = self.map._transform_pc(R_g2e, trans_g_e, self.map._transform_pc_inv(R_a2g, trans_g_a, obs))
                future = self.map._transform_pc(R_g2e, trans_g_e, self.map._transform_pc_inv(R_a2g, trans_g_a, future))

                preds_k = []
                for k in range(best_k):
                    pred = self.map._transform_pc(R_g2e, trans_g_e,
                                                 self.map._transform_pc_inv(R_a2g, trans_g_a, preds[k, :, :]))
                    preds_k.append(np.expand_dims(pred, axis=0))
                preds = np.concatenate(preds_k, axis=0)

                obs_traj.append(np.expand_dims(obs, axis=1))
                future_traj.append(np.expand_dims(future, axis=1))
                pred_traj_k.append(np.expand_dims(preds, axis=2))

        obs_traj = np.concatenate(obs_traj, axis=1)
        future_traj = np.concatenate(future_traj, axis=1)
        pred_traj_k = np.concatenate(pred_traj_k, axis=2)

        return obs_traj, future_traj, pred_traj_k

    def convert_agentcentric_to_global(self, _obs_traj, _future_traj, _pred_traj, agent_ids, agent_samples):

        best_k = _pred_traj.shape[0]
        num_agents = len(agent_samples)

        # extend dims
        z_axis = np.expand_dims(_future_traj[:, :, 2].reshape(self.pred_len, num_agents, 1), axis=0)
        z_axis = np.repeat(z_axis, best_k, axis=0)
        _pred_traj = np.concatenate([_pred_traj, z_axis], axis=3)

        obs_traj, future_traj, pred_traj_k = [], [], []
        for i in range(num_agents):

            # ego-vehicle
            R_a2g = agent_samples[i].R_a2g
            trans_g_a = agent_samples[i].trans_g

            obs = _obs_traj[:, i, :]
            future = _future_traj[:, i, :]
            preds = _pred_traj[:, :, i, :]

            obs = self.map._transform_pc_inv(R_a2g, trans_g_a, obs)
            future = self.map._transform_pc_inv(R_a2g, trans_g_a, future)

            preds_k = []
            for k in range(best_k):
                pred = self.map._transform_pc_inv(R_a2g, trans_g_a, preds[k, :, :])
                preds_k.append(np.expand_dims(pred, axis=0))
            preds = np.concatenate(preds_k, axis=0)

            obs_traj.append(np.expand_dims(obs, axis=1))
            future_traj.append(np.expand_dims(future, axis=1))
            pred_traj_k.append(np.expand_dims(preds, axis=2))

        obs_traj = np.concatenate(obs_traj, axis=1)
        future_traj = np.concatenate(future_traj, axis=1)
        pred_traj_k = np.concatenate(pred_traj_k, axis=2)

        return obs_traj, future_traj, pred_traj_k

    def crop_topview(self, map):


        axis_range = self.x_range[1] - self.x_range[0]
        scale = float(self.map_size - 1) / axis_range

        left_bot = np.array([self.x_crop_range[0], self.y_crop_range[1]]).reshape(1, 2)
        left_up = np.array([self.x_crop_range[1], self.y_crop_range[1]]).reshape(1, 2)
        right_bot = np.array([self.x_crop_range[0], self.y_crop_range[0]]).reshape(1, 2)
        right_up = np.array([self.x_crop_range[1], self.y_crop_range[0]]).reshape(1, 2)
        corners = np.concatenate([left_bot, left_up, right_bot, right_up], axis=0)

        col_pels = -(corners[:, 1] * scale).astype(np.int32)
        row_pels = -(corners[:, 0] * scale).astype(np.int32)

        col_pels += int(np.trunc(self.x_range[1] * scale))
        row_pels += int(np.trunc(self.x_range[1] * scale))

        return np.copy(map[row_pels[1]:row_pels[0], col_pels[0]:col_pels[2], :])

    def traverse_linked_list(self, obj, tablekey, direction, inclusive=False):
        return nuscenes_module.traverse_linked_list(self.nusc, obj, tablekey, direction, inclusive)




def _pooling_operation(img, trajectory, x_range, y_range, map_size):

    seq_len, dim = trajectory.shape
    axis_range_y = y_range[1] - y_range[0]
    axis_range_x = x_range[1] - x_range[0]
    scale_y = float(map_size - 1) / axis_range_y
    scale_x = float(map_size - 1) / axis_range_x


    col_pels = -(trajectory[:, 1] * scale_y).astype(np.int32)
    row_pels = -(trajectory[:, 0] * scale_x).astype(np.int32)

    col_pels += int(np.trunc(y_range[1] * scale_y))
    row_pels += int(np.trunc(x_range[1] * scale_x))

    values = []
    for j in range(0, seq_len):
        cur_c = col_pels[j]
        cur_r = row_pels[j]

        if (cur_c > -1 or cur_r > -1 or cur_c < map_size or cur_r < map_size):
            # current position is inside the drivable space
            if (img[cur_r, cur_c, 0] == 1):
                values.append(0)
            # current position is outside the drivable space
            else:
                values.append(1)
        # update, 220204
        # current position is outside the drivable space
        else:
            values.append(1)

    return np.array(values).reshape(1, seq_len)



def _draw_centerlines(img, cur_centerline, x_range, y_range, map_size, color=(255, 255, 255)):

    # for displaying images
    axis_range_y = y_range[1] - y_range[0]
    axis_range_x = x_range[1] - x_range[0]
    scale_y = float(map_size - 1) / axis_range_y
    scale_x = float(map_size - 1) / axis_range_x

    col_pels = -(cur_centerline[:, 1] * scale_y).astype(np.int32)
    row_pels = -(cur_centerline[:, 0] * scale_x).astype(np.int32)

    col_pels += int(np.trunc(y_range[1] * scale_y))
    row_pels += int(np.trunc(x_range[1] * scale_x))

    for j in range(1, cur_centerline.shape[0]):
        cv2.line(img, (col_pels[j], row_pels[j]), (col_pels[j - 1], row_pels[j - 1]), color, 2)

    return img

def _draw_bbox_on_topview(img, bbox, incolor, x_range, y_range, map_size):
    # for displaying images
    axis_range_y = y_range[1] - y_range[0]
    axis_range_x = x_range[1] - x_range[0]
    scale_y = float(map_size - 1) / axis_range_y
    scale_x = float(map_size - 1) / axis_range_x

    color = (255, 255, 0)
    if (incolor != None):
        color = incolor

    # to topview image domain
    col_pels = -(bbox[:, 1] * scale_y).astype(np.int32)
    row_pels = -(bbox[:, 0] * scale_x).astype(np.int32)

    col_pels += int(np.trunc(y_range[1] * scale_y))
    row_pels += int(np.trunc(x_range[1] * scale_x))

    cv2.line(img, (col_pels[6], row_pels[6]), (col_pels[2], row_pels[2]), color, 1)
    cv2.line(img, (col_pels[6], row_pels[6]), (col_pels[7], row_pels[7]), color, 1)
    cv2.line(img, (col_pels[3], row_pels[3]), (col_pels[2], row_pels[2]), color, 1)
    cv2.line(img, (col_pels[3], row_pels[3]), (col_pels[7], row_pels[7]), color, 1)

    cv2.line(img, (col_pels[7], row_pels[7]),
             (int((col_pels[3] + col_pels[2]) / 2), int((row_pels[3] + row_pels[2]) / 2)), (0, 255, 255), 1)
    cv2.line(img, (col_pels[6], row_pels[6]),
             (int((col_pels[3] + col_pels[2]) / 2), int((row_pels[3] + row_pels[2]) / 2)), color, 1)

    return img

def _draw_agent_trajectories(img, trajectory, x_range, y_range, map_size, category):

    seq_len, batch, dim = trajectory.shape
    axis_range_y = y_range[1] - y_range[0]
    axis_range_x = x_range[1] - x_range[0]
    scale_y = float(map_size - 1) / axis_range_y
    scale_x = float(map_size - 1) / axis_range_x

    for b in range(batch):

        if (category[b] == 0):
            circle_size = 5
        else:
            circle_size = 2

        col_pels = -(trajectory[:, b, 1] * scale_y).astype(np.int32)
        row_pels = -(trajectory[:, b, 0] * scale_x).astype(np.int32)

        col_pels += int(np.trunc(y_range[1] * scale_y))
        row_pels += int(np.trunc(x_range[1] * scale_x))

        for j in range(0, seq_len):

            if (np.isnan(col_pels[j])):
                continue

            brightness = int(255.0 * float(j + 1) / float(seq_len + 1))
            color = (0, 0, brightness)
            img = cv2.circle(img, (col_pels[j], row_pels[j]), circle_size, color, -1)

    return img
