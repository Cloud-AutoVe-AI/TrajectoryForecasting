from NuscenesDataset.visualization import *
from NuscenesDataset.map import Map
from NuscenesDataset.preprocess import DatasetBuilder
import NuscenesDataset.nuscenes.nuscenes as nuscenes_module
from torch.utils.data import Dataset
from utils.functions import read_config


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

        self.model_mode = args.model_mode
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
        self.is_data_conv = args.is_data_conv

        # checks existance of dataset file and create
        self.save_path = config['Nuscenes']['preproc_dataset_path'] + '/%dsec_%dsec' \
                    % (args.past_horizon_seconds, args.future_horizon_seconds)
        if (os.path.exists(self.save_path)==False):
            os.mkdir(self.save_path)
        file_name = 'nuscenes_%s_cat%d.cpkl' % (exp_type, args.category_filtering_method)

        builder = DatasetBuilder(args, map=self.map, isTrain=isTrain)
        if (os.path.exists(os.path.join(self.save_path, file_name))==False):
            builder.make_preprocessed_data(os.path.join(self.save_path, file_name), exp_type)

        # load dataset file
        with open(os.path.join(self.save_path, file_name), 'rb') as f:
            dataset = dill.load(f, encoding='latin1')
            print(">> {%s} is loaded .." % (os.path.join(self.save_path, file_name)))

        self.train_data = dataset[0]
        self.valid_data = dataset[1]
        self.test_data = dataset[2]

        if (args.is_data_conv == 0):
            self.train_data, self.valid_data, self.test_data = [], [], []
            for _, scene in enumerate(tqdm(dataset[0], desc='refactoring train data')):
                self.train_data += self.extract_data_from_scene(scene, isTrain=True)
                if (args.random_rotation == 1):
                    self.train_data += self.extract_data_from_scene(scene, isTrain=True, isAug=True)
            for _, scene in enumerate(tqdm(dataset[1], desc='refactoring valid data')):
                self.valid_data += self.extract_data_from_scene(scene)
            for _, scene in enumerate(tqdm(dataset[2], desc='refactoring test data')):
                self.test_data += self.extract_data_from_scene(scene)

        self.num_train_scenes = len(self.train_data)
        self.num_valid_scenes = len(self.valid_data)
        self.num_test_scenes = len(self.test_data)

        print(">> Loader is loaded from {%s} " % os.path.basename(__file__))



    def __len__(self):
        return len(self.train_data)


    def __getitem__(self, idx):

        # TODO : training with nuscenes dataset is not supported in this version !!

        # current scene data
        obs_traj, future_traj, obs_traj_e, future_traj_e, R_map, R_traj, degree, num_agents, agent_id, scene = self.train_data[idx]

        # topview image
        scene_location = scene.city_name
        lidar_now_token = scene.lidar_token_seq[self.obs_len-1]
        lidar_now_data = self.nusc.get('sample_data', lidar_now_token)
        ego_pose = self.nusc.get('ego_pose', lidar_now_data['ego_pose_token'])
        feature_topview = self.map.make_topview_map_for_pedestrian(ego_pose, scene_location, self.x_range,
                                                                    self.y_range, self.map_size)


        if (degree is not None):
            img = (255.0 * np.copy(feature_topview)).astype('uint8')
            height, width, channel = img.shape
            matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 360-degree, 1)
            feature_topview = cv2.warpAffine(img, matrix, (width, height)).astype('float') / 255.0


        # # debug -------------
        # img = (255.0 * np.copy(feature_topview)).astype('uint8')
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
        #         img = draw_bbox_on_topview(img, agent.trajectory[self.obs_len-1, 1:3].reshape(1, 2), R_map[_], agent.bbox(),
        #                                    self.x_range, self.y_range, self.map_size, (0, 0, 255))
        #
        # feature_topview = np.copy(img).astype('float')
        # cv2.imshow("", img)
        # cv2.waitKey(0)
        # # debug -------------

        # to Tensor
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
            data = self.valid_data[index]
        else:
            data = self.test_data[index]

        # current scene data
        obs_traj, future_traj, obs_traj_e, future_traj_e, R_map, R_traj, degree, num_agents, agent_id, scene = data

        # topview image
        scene_location = scene.city_name
        lidar_now_token = scene.lidar_token_seq[self.obs_len-1]
        lidar_now_data = self.nusc.get('sample_data', lidar_now_token)
        ego_pose = self.nusc.get('ego_pose', lidar_now_data['ego_pose_token'])
        feature_topview = self.map.make_topview_map_for_pedestrian(ego_pose, scene_location, self.x_range,
                                                                    self.y_range, self.map_size)

        # debug -------------
        # img = (255.0 * np.copy(feature_topview)).astype('uint8')
        #
        # # height, width, channel = img.shape
        # # matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 360-degree, 1)
        # # img = cv2.warpAffine(img, matrix, (width, height))
        #
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
        #         img = draw_bbox_on_topview(img, agent.trajectory[self.obs_len-1, 1:3].reshape(1, 2), R_map[_], agent.bbox(),
        #                                    self.x_range, self.y_range, self.map_size, (0, 0, 255))
        #
        # cv2.imshow("", img)
        # cv2.waitKey(0)

        # grid_size = 0.1
        # num_grid = 400
        # x = create_ROI(grid_size, num_grid).reshape(num_grid ** 2, 2)
        # for a in range(0, obs_traj.shape[1]):
        #
        #     a_id = agent_id[0, a]
        #     a_token = scene.id_2_token_lookup[a_id]
        #
        #     cur_pos = obs_traj[-1, a, :2].reshape(1, 2)
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

        # return
        Ht, Hb = None, None
        return obs_traj, future_traj, obs_traj_e, future_traj_e, feature_topview, R_map, R_traj, Ht, Hb, num_agents, agent_id, scene


    def extract_data_from_scene(self, scene, isTrain=False, isAug=False):

        scene_location = scene.city_name
        lidar_now_token = scene.lidar_token_seq[self.obs_len-1]
        lidar_now_data = self.nusc.get('sample_data', lidar_now_token)
        ego_pose = self.nusc.get('ego_pose', lidar_now_data['ego_pose_token'])

        # trajectory ---------
        num_total_agents = scene.num_agents
        agent_id = np.zeros(shape=(1, num_total_agents))
        trajectories = np.full(shape=(self.obs_len+self.pred_len, num_total_agents, 3), fill_value=np.nan)
        category = np.zeros(shape=(1, num_total_agents)) # 1: pedestrian, 0:vehicle

        for idx, track_id in enumerate(scene.agent_dict):
            agent_id[0, idx] = scene.agent_dict[track_id].agent_id
            trajectory = scene.agent_dict[track_id].trajectory
            trajectories[:, idx, :] = trajectory[:, 1:]

            # ego should be included
            if ('pedestrian' in scene.agent_dict[track_id].type):
                category[0, idx] = 1

        # find agents inside the limit range
        limit_range = self.limit_range
        if (np.random.rand(1) < self.limit_range_change_prob and isTrain):
            limit_range = float(random.randint(15, int(self.limit_range)))

        valid_flag = np.sqrt(np.sum(trajectories[self.obs_len-1, :, :2] ** 2, axis=1)) < limit_range
        if (np.count_nonzero(valid_flag) < 1):
            return []
        trajectories = trajectories[:, valid_flag, :]
        agent_id = agent_id[:, valid_flag]
        category = category[:, valid_flag]


        # find agents who has valid observation
        valid_flag = np.sum(trajectories[self.obs_len-self.min_obs_len:self.obs_len, :, 0], axis=0) > -1000
        if (np.count_nonzero(valid_flag) < 1):
            return []
        trajectories = trajectories[:, valid_flag, :]
        agent_id = agent_id[:, valid_flag]
        category = category[:, valid_flag]


        # find trajectories that have full future traj,
        valid_flag = np.min(trajectories[self.obs_len:, :, 0], axis=0) > -1000
        if (np.count_nonzero(valid_flag) < 1):
            return []
        trajectories = trajectories[:, valid_flag, :]
        agent_id = agent_id[:, valid_flag]
        category = category[:, valid_flag]


        # remove 'vehicle' except for 'ego vehicle'
        valid_flag = (category == 1)[0]
        if (np.count_nonzero(valid_flag) < 1):
            return []


        feature_topview = self.map.make_topview_map_loadertypeE(ego_pose, scene_location, self.x_range, self.y_range,
                                                                self.map_size, trajectories[:self.obs_len], category[0])

        # remove 'vehicle' except for 'ego vehicle'
        trajectories = trajectories[:, valid_flag, :]
        agent_id = agent_id[:, valid_flag]
        num_agents = trajectories.shape[1]


        # split into observation and future
        obs_traj = trajectories[:self.obs_len]
        future_traj = trajectories[self.obs_len:]


        # rotation matrix, note : pedestrian has eye(2) rotation matrix)
        R_map = np.concatenate([np.expand_dims(np.eye(2), axis=0) for _ in range(num_agents)], axis=0)
        R_traj = np.copy(R_map)

        # conversion to ego-centric coordinate system
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

        # 'nan' in ego-centric trajectories becomes zero
        nan_pos_obs = np.isnan(obs_traj[:, :, 2])
        nan_pos_future = np.isnan(future_traj[:, :, 2])
        obs_traj_e[nan_pos_obs] = 0
        future_traj_e[nan_pos_future] = 0

        # reshape
        obs_traj = obs_traj.reshape(self.obs_len, num_agents, 3)
        future_traj = future_traj.reshape(self.pred_len, num_agents, 3)
        obs_traj_e = obs_traj_e.reshape(self.obs_len, num_agents, 2)
        future_traj_e = future_traj_e.reshape(self.pred_len, num_agents, 2)
        R_map = R_map.reshape(num_agents, 2, 2)
        R_traj = R_traj.reshape(num_agents, 2, 2)
        agent_id = agent_id.reshape(1, num_agents)

        return [obs_traj, future_traj, obs_traj_e, future_traj_e, feature_topview, R_map, R_traj, num_agents, agent_id, scene]


    def traverse_linked_list(self, obj, tablekey, direction, inclusive=False):
        return nuscenes_module.traverse_linked_list(self.nusc, obj, tablekey, direction, inclusive)


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

# update, 220210
def dataset_conversion(args, accept_ratio=1.0):

    from utils.dataset_conversion import Dtype_Scene as Scene

    part_size = 1000
    pidx = 0

    loader = DatasetLoader(args=args, isTrain=True, dtype=torch.FloatTensor)
    num_scenes = len(loader.train_data)
    for ps in range(0, num_scenes, part_size):
        start = ps
        end = ps + part_size
        if (end > num_scenes):
            end = num_scenes
        target_data = loader.train_data[start:end]

        scenes = []
        for i, scene in enumerate(tqdm(target_data, desc='train data conversion')):

            if (np.random.rand(1) > accept_ratio):
                continue

            data = loader.extract_data_from_scene(scene, isTrain=False)
            if (len(data) == 0):
                continue

            obs_traj, future_traj, obs_traj_a, future_traj_a, feature_topview, R_map, R_traj, num_agents, agent_id, _ = data

            # debug -------------
            # img = (255.0 * np.copy(feature_topview[:, :, :3])).astype('uint8')
            # overall_trajs = np.concatenate([obs_traj[:, :, :2], future_traj[:, :, :2]], axis=0)
            # for _, id in enumerate(agent_id[0, :]):
            #     agent_key = scene.id_2_token_lookup[int(id)]
            #     agent = scene.agent_dict[agent_key]
            #
            #     if (agent.agent_id in agent_id[0, :].tolist()):
            #
            #         img = draw_traj_on_topview(img, overall_trajs[:, np.argwhere(agent.agent_id == agent_id[0]).item(), :2],
            #                                    loader.obs_len, loader.x_range, loader.y_range, loader.map_size, (0, 255, 255))
            #
            #         img = draw_bbox_on_topview(img, agent.trajectory[loader.obs_len-1, 1:3].reshape(1, 2), R_map[_], agent.bbox(),
            #                                    loader.x_range, loader.y_range, loader.map_size, (0, 0, 255))
            #
            # cv2.imshow("", img)
            # cv2.waitKey(0)
            #
            # grid_size = 0.1
            # num_grid = 400
            # x = create_ROI(grid_size, num_grid).reshape(num_grid ** 2, 2)
            # for a in range(0, obs_traj.shape[1]):
            #
            #     a_id = agent_id[0, a]
            #     a_token = scene.id_2_token_lookup[a_id]
            #
            #     cur_pos = obs_traj[-1, a, :2].reshape(1, 2)
            #     cur_R = R_map[a]
            #
            #     x_rot = np.matmul(cur_R, x.T).T + cur_pos
            #     aa = pooling_operation(img, x_rot, loader.x_range, loader.y_range, loader.map_size)
            #     b = aa.reshape(num_grid, num_grid, 3).astype('uint8')
            #     b = cv2.resize(b, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            #
            #     obs_traj_ego = np.matmul(R_traj[a], (overall_trajs[:, a, :2]-cur_pos).T).T
            #     c = np.zeros(shape=(2*num_grid, 2*num_grid, 3))
            #     c = draw_traj_on_topview(c, obs_traj_ego, loader.obs_len, (-40, 40), (-40, 40), 2*num_grid, (0, 255, 255))
            #
            #     d = np.hstack([b, c])
            #     cv2.imshow('test', d.astype('uint8'))
            #     cv2.waitKey(0)
            # debug -------------

            _scene = Scene(dataset='nuscenes', data_info=args, num_agents=num_agents)
            _scene.obs_traj = obs_traj
            _scene.future_traj = future_traj
            _scene.obs_traj_a = obs_traj_a
            _scene.future_traj_a = future_traj_a
            _scene.feature_topview = (255*feature_topview).astype('uint8')
            _scene.R_map = R_map
            _scene.R_traj = R_traj
            _scene.agent_id = agent_id
            scenes.append(_scene)

        save_path = '/home/dooseop/DATASET/voss/nuscenes'
        file_name = 'nuscenes_typeDconv_%ds%ds_part%d_ped.cpkl' % (args.past_horizon_seconds, args.future_horizon_seconds, pidx)
        with open(os.path.join(save_path, file_name), 'wb') as f:
            dill.dump(scenes, f, protocol=dill.HIGHEST_PROTOCOL)
        print('>> {%s} is created .. ' % os.path.join(save_path, file_name))
        pidx += 1

if __name__ == '__main__':

    abspath = os.path.dirname(os.path.realpath(__file__))
    os.chdir(Path(abspath).parent.absolute())

    # model info
    dataset_type = 'nuscenes'
    model_name = 'autove_ped'
    exp_id = 300

    # path to saved network
    folder_name = dataset_type + '_' + model_name + '_model' + str(exp_id)
    path = os.path.join('./saved_models/', folder_name)

    # load parameter setting
    with open(os.path.join(path, 'config.pkl'), 'rb') as f:
        args = pickle.load(f)
    args.dataset_path = "/home/dooseop/DATASET/nuscenes/"
    args.category_filtering_method = 2
    args.is_data_conv = 1
    print_training_info(args)

    dataset_conversion(args, accept_ratio=0.5)
