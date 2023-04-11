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
        self.lidar_map_ch_dim = args.lidar_map_ch_dim
        self.num_lidar_sweeps = args.num_lidar_sweeps

        self.is_crop_topview = args.is_crop_topview
        self.x_crop_range = (args.x_crop_min, args.x_crop_max)
        self.y_crop_range = (args.y_crop_min, args.y_crop_max)
        self.is_best_lane_only = args.is_best_lane_only
        self.is_random_path_order = args.is_random_path_order
        # update, 220103
        self.is_repeat_scene = args.is_repeat_scene



        self.path_resol = args.path_resol
        self.path_len_f = args.max_path_len_forward
        self.path_len_b = args.max_path_len_backward
        self.num_pos_f = int(args.max_path_len_forward / self.path_resol)
        self.num_pos_b = int(args.max_path_len_backward / self.path_resol)
        self.num_max_paths = args.num_max_paths


        self.use_preprocessed_lidar = args.use_preprocessed_lidar
        self.voxel_save_dir = config['Nuscenes']['voxel_dir']


        self.stop_agents_remove_prob = args.stop_agents_remove_prob
        self.limit_range_change_prob = args.limit_range_change_prob


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
        obs_traj_ta, future_traj_ta, obs_traj_ngh, future_traj_ngh, map, num_neighbors, valid_neighbor, possible_paths, \
            lane_label = self.extract_data_from_scene(self.train_data[idx])

        obs_traj_ta = torch.from_numpy(obs_traj_ta).type(self.dtype)
        future_traj_ta = torch.from_numpy(future_traj_ta).type(self.dtype)
        obs_traj_ngh = torch.from_numpy(obs_traj_ngh).type(self.dtype)
        future_traj_ngh = torch.from_numpy(future_traj_ngh).type(self.dtype)
        map = torch.from_numpy(map).permute(2, 0, 1).type(self.dtype)
        map = torch.unsqueeze(map, dim=0)
        possible_paths = torch.from_numpy(possible_paths).type(self.dtype)
        lane_label = torch.from_numpy(lane_label).type(self.dtype)

        return obs_traj_ta, future_traj_ta, obs_traj_ngh, future_traj_ngh, \
               map, num_neighbors, valid_neighbor, possible_paths, lane_label

    def next_sample(self, index, mode):

        # current scene data
        if (mode == 'valid'):
            obs_traj_ta, future_traj_ta, obs_traj_ngh, future_traj_ngh, map, num_neighbors, valid_neighbor, possible_paths, \
                lane_label = self.extract_data_from_scene(self.valid_data[index])
            return obs_traj_ta, future_traj_ta, obs_traj_ngh, future_traj_ngh, \
                   map, num_neighbors, valid_neighbor, possible_paths, lane_label
        else:
            scene = self.test_data[index]
            agent_samples = self.refactoring(scene)

            obs_traj, future_traj, obs_traj_ngh, future_traj_ngh, map, num_neighbors, valid_neighbor, agent_ids, possible_paths,\
                lane_label = [], [], [], [], [], [], [], [], [], []

            for i in range(len(agent_samples)):

                # agent_index
                agent_id = agent_samples[i].target_agent_index

                # current sample
                _obs_traj, _future_traj, _obs_traj_ngh, _future_traj_ngh, _map, _num_neighbors, _valid_neighbor, _possible_paths,\
                    _lane_label = self.extract_data_from_scene(agent_samples[i])

                _obs_traj = torch.from_numpy(_obs_traj).type(self.dtype)
                _future_traj = torch.from_numpy(_future_traj).type(self.dtype)
                _obs_traj_ngh = torch.from_numpy(_obs_traj_ngh).type(self.dtype)
                _future_traj_ngh = torch.from_numpy(_future_traj_ngh).type(self.dtype)
                _map = torch.from_numpy(_map).permute(2, 0, 1).type(self.dtype)
                _map = torch.unsqueeze(_map, dim=0)
                _possible_paths = torch.from_numpy(_possible_paths).type(self.dtype)
                _lane_label = torch.from_numpy(_lane_label).type(self.dtype)

                obs_traj.append(_obs_traj)
                future_traj.append(_future_traj)
                obs_traj_ngh.append(_obs_traj_ngh)
                future_traj_ngh.append(_future_traj_ngh)
                map.append(_map)
                num_neighbors.append(_num_neighbors)
                valid_neighbor.append(_valid_neighbor)
                agent_ids.append(agent_id)
                possible_paths.append(_possible_paths)
                lane_label.append(_lane_label)

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
            possible_paths = torch.cat(possible_paths, dim=1)
            lane_label = torch.cat(lane_label, dim=0)


            return [obs_traj, future_traj, obs_traj_ngh, future_traj_ngh, map, seq_start_end, valid_neighbor,
                    possible_paths, lane_label, agent_ids, agent_samples, scene]


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
        yaw_g = np.full(shape=(self.obs_len + self.pred_len, num_total_agents, 1), fill_value=np.nan) # update, 220103
        for idx, track_id in enumerate(scene.agent_dict):
            agent_ids[0, idx] = scene.agent_dict[track_id].agent_id
            trajectory = scene.agent_dict[track_id].trajectory
            trajectories[:, idx, :] = trajectory[:, 1:]

            # update, 220103
            yaw_g[:, idx, :] = scene.agent_dict[track_id].yaw_global

        # find agents inside the limit range
        valid_flag = np.sqrt(np.sum(trajectories[self.obs_len-1, :, :2] ** 2, axis=1)) < self.limit_range
        trajectories = trajectories[:, valid_flag, :]
        yaw_g = yaw_g[:, valid_flag, :] # update, 220103
        agent_ids = agent_ids[:, valid_flag]

        # find agents who has valid observation
        valid_flag = np.sum(trajectories[self.obs_len - self.min_obs_len:self.obs_len, :, 0], axis=0) > -1000
        trajectories = trajectories[:, valid_flag, :]
        yaw_g = yaw_g[:, valid_flag, :] # update, 220103
        agent_ids = agent_ids[:, valid_flag]

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

            # update, 220103
            # bboxes_g = np.copy(bboxes)
            # bboxes_g[a, :, :] = self.map._transform_pc_inv(R_a2g, trans_a, bboxes[a, :, :])

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
            agent_sample.trajectories_g = trajectories_g # update, 220103
            agent_sample.yaw_g = yaw_g  # update, 220103
            agent_sample.R_a2g = R_a2g
            agent_sample.R_g2a = R_g2a
            agent_sample.trans_g = trans_a
            agent_sample.agent_ids = agent_ids
            agent_sample.possible_lanes = agent.possible_lanes
            samples.append(agent_sample)

            if (isTrain and self.is_repeat_scene == 1 and len(agent.possible_lanes) > 0):
                idx = np.argwhere(agent_ids[0, :] == agent_sample.target_agent_index)[0][0]
                rt = self.check_centerline_curvature(trajectories_a[:, idx, :], [agent.possible_lanes[0]], R_g2a, trans_a, scene.city_name)
                if (rt):
                    samples.append(agent_sample)

            # debug ----
            # cv2.imshow("", img.astype('uint8'))
            # cv2.waitKey(0)
            # debug ----


        return samples

    def extract_data_from_scene(self, scene, isTrain=True):

        agent_ids = scene.agent_ids
        target_agent_index = scene.target_agent_index
        trajectories = scene.trajectories
        # bboxes = scene.bboxes update, 220104
        possible_lanes = scene.possible_lanes

        R_g2a = scene.R_g2a
        R_a2g = scene.R_a2g
        trans_g = scene.trans_g

        # find agents inside the limit range
        valid_flag = np.sqrt(np.sum(trajectories[self.obs_len-1, :, :2] ** 2, axis=1)) < self.limit_range
        trajectories = trajectories[:, valid_flag, :]
        agent_ids = agent_ids[:, valid_flag]
        # bboxes_g = bboxes[valid_flag, :, :] update, 220104

        # split into target agent and neighbors
        idx = np.argwhere(agent_ids[0, :] == target_agent_index)[0][0]

        num_agents = agent_ids.shape[1]
        trajectory_ta = np.full(shape=(self.obs_len+self.pred_len, 1, 3), fill_value=np.nan)
        trajectories_ngh = []
        for l in range(num_agents):
            if (l == idx):
                trajectory_ta[:, :, :] = np.copy(trajectories[:, l, :].reshape(self.obs_len+self.pred_len, 1, 3))
            else:
                trajectories_ngh.append(np.copy(trajectories[:, l, :].reshape(self.obs_len+self.pred_len, 1, 3)))

        num_neighbors = len(trajectories_ngh)
        if (num_neighbors == 0):
            trajectories_ngh = np.full(shape=(self.obs_len+self.pred_len, 1, 3), fill_value=np.nan)
            valid_neighbor = False
            num_neighbors += 1
        else:
            trajectories_ngh = np.concatenate(trajectories_ngh, axis=1)
            valid_neighbor = True


        # calc speed and heading
        trajectory_ta_ext = self.calc_speed_heading(trajectory_ta)
        trajectories_ngh_ext = []
        for n in range(trajectories_ngh.shape[1]):
            trajectories_ngh_ext.append(self.calc_speed_heading(trajectories_ngh[:, n, :].reshape(self.obs_len+self.pred_len, 1, 3)))
        trajectories_ngh_ext = np.concatenate(trajectories_ngh_ext, axis=1)

        # split into observation and future
        obs_traj_ta = np.copy(trajectory_ta_ext[:self.obs_len])
        future_traj_ta = np.copy(trajectory_ta_ext[self.obs_len:])
        obs_traj_ngh = np.copy(trajectories_ngh_ext[:self.obs_len])
        future_traj_ngh = np.copy(trajectories_ngh_ext[self.obs_len:])

        # remove 'nan' in observation
        obs_traj_ta = self.remove_nan(obs_traj_ta)
        obs_traj_ngh = self.remove_nan(obs_traj_ngh)

        # draw topview map
        if (self.is_best_lane_only == 1 and len(possible_lanes)>0):
            target_lanes = [possible_lanes[0]]
        else:
            target_lanes = copy.deepcopy(possible_lanes)

        # map = self.map.make_topview_map_agent_centric(target_lanes, trans_g[0], R_a2g, R_g2a, scene.city_name,
        #                                               self.x_range, self.y_range, self.map_size)

        # if (self.is_crop_topview):
        #     map = self.crop_topview(map)
        map = np.zeros(shape=(10, 10, 3))

        # NOTE : must shuffle the order of the possible lanes
        possible_paths, lane_label = self.get_lane_coords(possible_lanes, R_g2a, trans_g, scene.city_name)

        # # debug ---
        # # img = (255 * map).astype('uint8')
        # img = np.sum(map, axis=2).reshape(self.map_size, self.map_size, 1).repeat(3, axis=2).astype('float')
        # img = 128 * img / np.max(img)
        #
        # for aa in range(num_agents):
        #
        #     if (agent_ids[0, aa] == target_agent_index):
        #
        #         for ll in range(possible_paths.shape[1]):
        #             if (np.count_nonzero(np.isnan(possible_paths[:, ll, :])) == 0):
        #                 img = _draw_centerlines(img.astype('uint8'), possible_paths[:, ll, :], self.x_range, self.y_range, self.map_size, color=(0, 255, 0))
        #
        #
        #         img = _draw_centerlines(img.astype('uint8'), trajectories[:, aa, :], self.x_range, self.y_range,
        #                                 self.map_size)
        #         bbox = self.map._transform_pc(R_g2a, trans_g, bboxes_g[aa, :, :])
        #         img = _draw_bbox_on_topview(img.astype('uint8'), bbox, (0, 0, 255), self.x_range, self.y_range, self.map_size)
        #
        #
        #
        #
        # # img = self.crop_topview(img)
        # cv2.imshow("", img.astype('uint8'))
        # cv2.waitKey(0)
        # # debug ---

        # assert (np.count_nonzero(np.isnan(obs_traj_ta)) ==0)
        # assert (np.count_nonzero(np.isnan(future_traj_ta)) == 0)
        # assert (np.count_nonzero(np.isnan(obs_traj_ngh)) == 0)

        return obs_traj_ta, future_traj_ta, obs_traj_ngh, future_traj_ngh, map, num_neighbors, valid_neighbor, \
               possible_paths, lane_label

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

    # update, 220103
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

    def get_lane_coords(self, possible_lanes, R_g2a, trans_g, location):

        # TODO : lane should be descretize equally spaced
        filter = np.array([0.5, 0, 0.5])
        target_spacing = np.arange(0, self.path_len_f, self.path_resol)
        min_path_len = 5.0 # meter

        # get lane coordinates
        possible_paths = []
        for _, tok_seq in enumerate(possible_lanes):
            path = []

            # discretize and global2ego transform
            for __, tok in enumerate(tok_seq):
                lane_record = self.map.nusc_maps[location].get_arcline_path(tok)
                coords = np.array(discretize_lane(lane_record, resolution_meters=0.05))
                path.append(coords)
            path = np.concatenate(path, axis=0)
            path_agent_centric = self.map._transform_pc(R_g2a, trans_g, path)[:, :2]

            # find target segment, update, 211014
            start_idx = np.argmin(np.sum(np.abs(path_agent_centric[:, :2]), axis=1))
            path_agent_centric = path_agent_centric[start_idx:]
            path_len = path_agent_centric.shape[0]
            if (path_len < int(min_path_len/self.path_resol)):
                continue

            # sample equally-spaced
            point_dist = np.zeros(shape=(path_agent_centric.shape[0]))
            point_dist[1:] = np.sqrt(np.sum((path_agent_centric[1:] - path_agent_centric[:-1])**2, axis=1))
            sorted_index = np.searchsorted(np.cumsum(point_dist), target_spacing, side='right')
            chk = sorted_index < path_len
            sorted_index = sorted_index[chk]
            path_agent_centric = path_agent_centric[sorted_index]

            # centerline quality
            seq_len = path_agent_centric.shape[0]
            point_dist = self.path_resol * np.ones(shape=seq_len)
            point_dist[1:] = np.sqrt(np.sum((path_agent_centric[1:] - path_agent_centric[:-1]) ** 2, axis=1))

            # update, 211015 (b/o exp 361)
            if (np.max(point_dist) > 1.1*self.path_resol or np.min(point_dist) < 0.9*self.path_resol):
                path_agent_centric_x_avg = np.convolve(path_agent_centric[:, 0], filter, mode='same').reshape(seq_len, 1)
                path_agent_centric_y_avg = np.convolve(path_agent_centric[:, 1], filter, mode='same').reshape(seq_len, 1)
                path_agent_centric_avg = np.concatenate([path_agent_centric_x_avg, path_agent_centric_y_avg], axis=1)

                chk = point_dist > 1.1*self.path_resol
                path_agent_centric[chk] = path_agent_centric_avg[chk]

                chk = point_dist < 0.9*self.path_resol
                path_agent_centric[chk] = path_agent_centric_avg[chk]

            # length of the segment
            path_len = path_agent_centric.shape[0]
            if (path_len < int(min_path_len/self.path_resol)):
                path_agent_centric = np.full(shape=(self.num_pos_f, 2), fill_value=np.nan) # update, 211015 (b/o exp363)
                path_len = path_agent_centric.shape[0]

            # path_len = path_agent_centric.shape[0]
            # if (path_len < int(min_path_len/self.path_resol)):
            #     continue

            if (path_len < self.num_pos_f):
                num_repeat = self.num_pos_f - path_len
                delta = (path_agent_centric[1:] - path_agent_centric[:-1])[-1].reshape(1, 2)
                delta = np.repeat(delta, num_repeat, axis=0)
                delta[0, :] += path_agent_centric[-1]
                padd = np.cumsum(delta, axis=0)
                path_agent_centric = np.concatenate([path_agent_centric, padd], axis=0)

                # # debug ---
                # plt.plot(path_agent_centric[:,0], path_agent_centric[:, 1])
                # plt.show()
                # # debug ---

            possible_paths.append(np.expand_dims(path_agent_centric, axis=1))
            assert(path_agent_centric.shape[0] == self.num_pos_f)

        num_repeat = 0
        if (self.num_max_paths > len(possible_paths)):
            num_repeat = self.num_max_paths - len(possible_paths)
            for i in range(num_repeat):
                possible_paths.append(np.full(shape=(self.num_pos_f, 1, 2), fill_value=np.nan))

        indices = [idx for idx in range(self.num_max_paths)]

        if (self.is_random_path_order == 1):
            random.shuffle(indices)

        possible_paths_random = []
        for _, idx in enumerate(indices):
            possible_paths_random.append(possible_paths[idx])
        possible_paths_random = np.concatenate(possible_paths_random, axis=1)


        label = np.zeros(shape=(1, self.num_max_paths))
        if (num_repeat == self.num_max_paths):
            best_match_idx = indices[0]
        else:
            best_match_idx = np.argwhere(np.array(indices) == 0)[0][0]
        label[0, best_match_idx] = 1

        return possible_paths_random, label

    def check_centerline_curvature(self, trajectory, possible_lanes, R_g2a, trans_g, location):

        # if trajectory is too short, skip
        traj_dist = np.sqrt(np.sum(trajectory[-1, :2]**2))
        if (traj_dist < 5):
            return False

        # get lane coordinates
        path = []
        target_spacing = np.arange(0, self.path_len_f, self.path_resol)
        tok_seq = possible_lanes[0]
        for __, tok in enumerate(tok_seq):
            lane_record = self.map.nusc_maps[location].get_arcline_path(tok)
            coords = np.array(discretize_lane(lane_record, resolution_meters=0.05))
            path.append(coords)
        path = np.concatenate(path, axis=0)
        path_agent_centric = self.map._transform_pc(R_g2a, trans_g, path)[:, :2]

        # find target segment
        start_idx = np.argmin(np.sum(np.abs(path_agent_centric[:, :2]), axis=1))
        path_agent_centric = path_agent_centric[start_idx:]
        path_len = path_agent_centric.shape[0]

        # sample equally-spaced
        point_dist = np.zeros(shape=(path_agent_centric.shape[0]))
        point_dist[1:] = np.sqrt(np.sum((path_agent_centric[1:] - path_agent_centric[:-1])**2, axis=1))
        sorted_index = np.searchsorted(np.cumsum(point_dist), target_spacing, side='right')
        chk = sorted_index < path_len
        sorted_index = sorted_index[chk]
        path_agent_centric = path_agent_centric[sorted_index]

        # check length of the segment
        path_len = path_agent_centric.shape[0]
        if (path_len < 1.5 * traj_dist + 1.0):
            return False

        # calc. curvature
        target_path_seg = path_agent_centric[:int(1.5*traj_dist)]
        curvature = self.curvature(target_path_seg)

        # consider high curvature lane
        if (np.max(curvature) > 0.1):
            return True
            # plt.plot(-1.0 * target_path_seg[:, 1], target_path_seg[:, 0], 'ko-')
            # plt.plot(-1.0 * trajectory[:, 1], trajectory[:, 0], 'ro-')
            # plt.show()

        return False

    def curvature(self, path):

        x_t = np.gradient(path[:, 0])
        y_t = np.gradient(path[:, 1])

        xx_t = np.gradient(x_t)
        yy_t = np.gradient(y_t)

        denorm = (x_t * x_t + y_t * y_t) ** 1.5
        chk_nan = (denorm == 0)
        denorm[chk_nan] = 1.0

        curvature = np.abs(xx_t * y_t - x_t * yy_t) / denorm
        curvature[chk_nan] = 0

        # flag = False
        # if (np.count_nonzero(chk_nan) != 0):
        #     flag = True

        return curvature

    def traverse_linked_list(self, obj, tablekey, direction, inclusive=False):
        return nuscenes_module.traverse_linked_list(self.nusc, obj, tablekey, direction, inclusive)


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
