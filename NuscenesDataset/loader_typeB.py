from utils.functions import read_config
from NuscenesDataset.visualization import *
from NuscenesDataset.map import Map
from NuscenesDataset.preprocess import DatasetBuilder
from torch.utils.data import Dataset
import NuscenesDataset.nuscenes.nuscenes as nuscenes_module
from NuscenesDataset.nuscenes.eval.prediction.splits import get_prediction_challenge_split
from NuscenesDataset.nuscenes.prediction import PredictHelper
from NuscenesDataset.nuscenes.prediction.helper import convert_global_coords_to_local, convert_local_coords_to_global
from NuscenesDataset.nuscenes.utils.geometry_utils import transform_matrix

from NuscenesDataset.nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer
from NuscenesDataset.nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistory
from NuscenesDataset.nuscenes.prediction.input_representation.interface import InputRepresentation
from NuscenesDataset.nuscenes.prediction.input_representation.combinators import Rasterizer

class DatasetLoader(Dataset):

    def __init__(self, args, dtype, isTrain=True):

        # exp settings
        exp_type = 'train' if isTrain else 'test'

        # nuscenes api
        self.nusc = nuscenes_module.NuScenes(version='v1.0-trainval', dataroot=args.dataset_path, verbose=False)

        # nuscenes map api
        self.map = Map(args, self.nusc)

        # prediction helper
        self.helper = PredictHelper(self.nusc)

        static_layer_rasterizer = StaticLayerRasterizer(self.helper)
        agent_rasterizer = AgentBoxesWithFadedHistory(self.helper, seconds_of_history=1)
        self.mtp_input_representation = InputRepresentation(static_layer_rasterizer, agent_rasterizer, Rasterizer())


        # params
        config = read_config()

        self.dtype = dtype
        self.obs_len = int(args.past_horizon_seconds * args.target_sample_period)
        self.pred_len = int(args.future_horizon_seconds * args.target_sample_period)
        self.min_obs_len = int(args.min_past_horizon_seconds * args.target_sample_period)
        self.future_horizon_seconds = args.future_horizon_seconds
        self.limit_range = args.limit_range

        self.stop_agents_remove_prob = args.stop_agents_remove_prob
        self.limit_range_change_prob = args.limit_range_change_prob


        if (exp_type == 'train'):
            # instances
            self.instance_samples = get_prediction_challenge_split('train', dataroot=args.dataset_path)
            random.shuffle(self.instance_samples)
            num_valid_samples = int(len(self.instance_samples) * args.val_ratio)
            self.train_data = self.instance_samples[num_valid_samples:]
            self.valid_data = self.instance_samples[:num_valid_samples]

            self.num_train_scenes = len(self.train_data)
            self.num_valid_scenes = len(self.valid_data)
            print(">> num train/valid agents : %d / %d" % (len(self.train_data), len(self.valid_data)))
        else:

            # instances
            self.instance_samples = get_prediction_challenge_split('val', dataroot=args.dataset_path)

            # checks existance of dataset file and create
            save_path = config['Nuscenes']['preproc_dataset_path'] + '/%dsec_%dsec' % (args.past_horizon_seconds, args.future_horizon_seconds)
            if (os.path.exists(save_path) == False):
                os.mkdir(save_path)
            file_name = 'nuscenes_%s_cat%d.cpkl' % (exp_type, args.category_filtering_method) # update, 211005

            builder = DatasetBuilder(args, map=self.map, isTrain=isTrain)
            if (os.path.exists(os.path.join(save_path, file_name)) == False):
                builder.make_preprocessed_data(os.path.join(save_path, file_name), exp_type)

            # load dataset file
            with open(os.path.join(save_path, file_name), 'rb') as f:
                dataset = dill.load(f, encoding='latin1')
                print(">> {%s} is loaded .." % (os.path.join(save_path, file_name)))  # update, 211005

            self.test_data = dataset[2]
            self.num_test_scenes = len(self.test_data)
            print(">> num test scenes : %d" % self.num_test_scenes)

        print(">> Loader is loaded from {%s} " % os.path.basename(__file__))

    def __len__(self):
        return len(self.train_data)


    def __getitem__(self, idx):

        # current sample
        instance_token, sample_token = self.train_data[idx].split("_")

        # future trajectory in agent-centric coordinate system
        future_traj_e = self.helper.get_future_for_agent(instance_token, sample_token,
                                                         seconds=self.future_horizon_seconds, in_agent_frame=True)
        pred_len = future_traj_e.shape[0]
        future_traj_e = future_traj_e.reshape(pred_len, 1, 2)

        # velocity, accel, yaw rate
        velo = self.helper.get_velocity_for_agent(instance_token, sample_token)
        accel = self.helper.get_acceleration_for_agent(instance_token, sample_token)
        yr = self.helper.get_heading_change_rate_for_agent(instance_token, sample_token)
        agent_states = np.array([velo, accel, yr]).reshape(1, 3)

        # topview HDmap
        topview_map = self.mtp_input_representation.make_input_representation(instance_token, sample_token)
        topview_map = np.expand_dims(topview_map, axis=0)

        future_traj_e = torch.from_numpy(future_traj_e).type(self.dtype)
        topview_map = torch.from_numpy(topview_map).permute(0, 3, 1, 2).type(self.dtype)
        agent_states = torch.from_numpy(agent_states).type(self.dtype)

        return future_traj_e, topview_map, agent_states, self.pred_len

    def next_sample(self, index, mode):

        if (mode == 'valid'):

            instance_token, sample_token = self.valid_data[index].split("_")

            # future trajectory in agent-centric coordinate system
            future_traj_e = self.helper.get_future_for_agent(instance_token, sample_token,
                                                             seconds=self.future_horizon_seconds, in_agent_frame=True)
            pred_len = future_traj_e.shape[0]
            future_traj_e = future_traj_e.reshape(pred_len, 1, 2)

            # velocity, accel, yaw rate
            velo = self.helper.get_velocity_for_agent(instance_token, sample_token)
            accel = self.helper.get_acceleration_for_agent(instance_token, sample_token)
            yr = self.helper.get_heading_change_rate_for_agent(instance_token, sample_token)
            agent_states = np.array([velo, accel, yr]).reshape(1, 3)

            # topview HDmap
            topview_map = self.mtp_input_representation.make_input_representation(instance_token, sample_token)
            topview_map = np.expand_dims(topview_map, axis=0)

            future_traj_e = torch.from_numpy(future_traj_e).type(self.dtype)
            topview_map = torch.from_numpy(topview_map).permute(0, 3, 1, 2).type(self.dtype)
            agent_states = torch.from_numpy(agent_states).type(self.dtype)

            return future_traj_e, topview_map, agent_states, self.pred_len

        else:
            scene = self.test_data[index]

            obs_traj, future_traj, future_traj_e, topview_map, agent_states, agent_ids, scene, valid_scene_flag\
                = self.extract_data_from_scene(scene)
            return obs_traj, future_traj, future_traj_e, topview_map, agent_states, agent_ids, scene, valid_scene_flag

    def extract_data_from_scene(self, scene):

        # get ego-pose
        lidar_now_token = scene.lidar_token_seq[self.obs_len-1]
        lidar_now_data = self.nusc.get('sample_data', lidar_now_token)
        ego_pose = self.nusc.get('ego_pose', lidar_now_data['ego_pose_token'])
        ego_from_global = transform_matrix(ego_pose['translation'],
                                           pyquaternion.Quaternion(ego_pose['rotation']), inverse=True)

        # trajectory
        num_total_agents = scene.num_agents
        agent_id = np.zeros(shape=(1, num_total_agents))
        trajectories = np.full(shape=(self.obs_len+self.pred_len, num_total_agents, 3), fill_value=np.nan)
        for idx, track_id in enumerate(scene.agent_dict):
            agent_id[0, idx] = scene.agent_dict[track_id].agent_id
            trajectory = scene.agent_dict[track_id].trajectory
            trajectories[:, idx, :] = trajectory[:, 1:]


        # find agents inside the limit range
        valid_flag = np.sqrt(np.sum(trajectories[self.obs_len-1, :, :2] ** 2, axis=1)) < self.limit_range
        trajectories = trajectories[:, valid_flag, :]
        agent_id = agent_id[:, valid_flag]

        # find agents who has valid observation
        valid_flag = np.sum(trajectories[self.obs_len - self.min_obs_len:self.obs_len, :, 0], axis=0) > -1000
        trajectories = trajectories[:, valid_flag, :]
        agent_id = agent_id[:, valid_flag]

        # find trajectories that have full future traj,
        valid_flag = ~np.isnan(np.sum(trajectories[self.obs_len:, :, 0], axis=0))
        trajectories = trajectories[:, valid_flag, :]
        agent_id = agent_id[:, valid_flag]

        # consider agents inside 'self.instance_samples' list
        valid_flag = [False]
        instance_samples = ['None_None']
        sample_token = scene.sample_token
        num_agents = trajectories.shape[1]
        for a in range(0, num_agents):
            ann_token = scene.id_2_token_lookup[agent_id[0, a]]
            if (ann_token == 'EGO'):
                continue

            ann = self.nusc.get('sample_annotation', ann_token)
            current = ann['instance_token'] + '_' + sample_token
            instance_samples.append(current)
            if (current in self.instance_samples):
                valid_flag.append(True)
            else:
                valid_flag.append(False)

        if (True not in valid_flag):
            return [], [], [], [], [], [], [], False

        valid_flag = np.array(valid_flag)
        instance_samples = np.array(instance_samples)[valid_flag]
        trajectories = trajectories[:, valid_flag, :]
        agent_id = agent_id[:, valid_flag]

        # split into observation and future
        obs_traj = trajectories[:self.obs_len]


        # agent yaw, speed, accel and top-view image
        num_valid_agents = 0
        topview_map, future_traj, future_traj_e, agent_states, agent_ids = [], [], [], [], []
        for _, instance_sample in enumerate(instance_samples):

            # current agent
            instance_token, sample_token = instance_sample.split("_")

            # agent global 3d
            ann = self.helper.get_sample_annotation(instance_token, sample_token)
            sequence = self.helper._iterate(ann, self.future_horizon_seconds, 'next')

            agent_global_3d = np.array([r['translation'] for r in sequence])
            agent_local_2d = convert_global_coords_to_local(agent_global_3d[:, :2], ann['translation'], ann['rotation'])
            # agent_global_2d = convert_local_coords_to_global(agent_local_2d, ann['translation'], ann['rotation'])

            if (agent_local_2d.shape[0] < self.pred_len):
                continue

            velo = self.helper.get_velocity_for_agent(instance_token, sample_token)
            if (np.isnan(velo)):
                continue

            accel = self.helper.get_acceleration_for_agent(instance_token, sample_token)
            if (np.isnan(accel)):
                continue

            yr = self.helper.get_heading_change_rate_for_agent(instance_token, sample_token)
            if (np.isnan(yr)):
                continue

            hdmap = self.mtp_input_representation.make_input_representation(instance_token, sample_token)
            agent_ego_3d = self.map.transform_pc(ego_from_global, agent_global_3d.T).T

            future_traj.append(np.expand_dims(agent_ego_3d, axis=1))
            future_traj_e.append(np.expand_dims(agent_local_2d, axis=1))
            agent_states.append(np.array([velo, accel, yr]).reshape(1, 3))
            topview_map.append(np.expand_dims(hdmap, axis=0))
            agent_ids.append(np.array(agent_id[0, _]).reshape(1, 1))

            num_valid_agents += 1

        if (num_valid_agents == 0):
            return [], [], [], [], [], [], [], False

        topview_map = np.concatenate(topview_map, axis=0)
        future_traj = np.concatenate(future_traj, axis=1)
        future_traj_e = np.concatenate(future_traj_e, axis=1)
        agent_states = np.concatenate(agent_states, axis=0)
        agent_ids = np.concatenate(agent_ids, axis=1)

        return obs_traj, future_traj, future_traj_e, topview_map, agent_states, agent_ids, scene, True

    def convert_to_egocentric(self, future_traj, pred_trajs, agent_ids, scene):

        '''
        future_traj (seq_len x batch x 2) : agent-centric
        pred_trajs (best_k x seq_len x batch x 2) : agent-centric

        agent-centric -> global -> ego-centric
        '''

        future_traj_e, pred_trajs_e = [], []

        lidar_now_token = scene.lidar_token_seq[self.obs_len-1]
        lidar_now_data = self.nusc.get('sample_data', lidar_now_token)
        ego_pose = self.nusc.get('ego_pose', lidar_now_data['ego_pose_token'])
        ego_from_global = transform_matrix(ego_pose['translation'],
                                           pyquaternion.Quaternion(ego_pose['rotation']), inverse=True)

        best_k, _, num_agents, _ = pred_trajs.shape
        for _ in range(num_agents):
            ann_token = scene.id_2_token_lookup[agent_ids[_]]
            ann = self.nusc.get('sample_annotation', ann_token)
            sequence = self.helper._iterate(ann, self.future_horizon_seconds, 'next')
            agent_global_3d = np.array([r['translation'] for r in sequence]) # seq_len x 3

            future_traj_global_2d = convert_local_coords_to_global(future_traj[:, _, :2], ann['translation'], ann['rotation'])
            future_traj_global_3d = np.concatenate([future_traj_global_2d, agent_global_3d[:, -1].reshape(self.pred_len, 1)], axis=1)
            future_traj_e.append(np.expand_dims(self.map.transform_pc(ego_from_global, future_traj_global_3d.T).T, axis=1))

            pred_trajs_k = []
            for k in range(best_k):
                pred_traj_global_2d = convert_local_coords_to_global(pred_trajs[k, :, _, :], ann['translation'], ann['rotation'])
                pred_traj_global_3d = np.concatenate([pred_traj_global_2d, agent_global_3d[:, -1].reshape(self.pred_len, 1)], axis=1)
                pred_trajs_k.append(np.expand_dims(self.map.transform_pc(ego_from_global, pred_traj_global_3d.T).T, axis=0))

            pred_trajs_e.append(np.expand_dims(np.concatenate(pred_trajs_k, axis=0), axis=0))

        future_traj_e = np.concatenate(future_traj_e, axis=1) # seq_len x num_agents x 3
        pred_trajs_e = np.transpose(np.concatenate(pred_trajs_e, axis=0), (1, 2, 0, 3)) # best_k x seq_len x num_agent x 3

        return future_traj_e, pred_trajs_e