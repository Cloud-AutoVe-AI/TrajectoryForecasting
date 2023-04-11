from utils.functions import read_config, print_voxelization_progress
from utils.libraries import *
from NuscenesDataset.agent import Agent
from NuscenesDataset.scene import Scene

import NuscenesDataset.nuscenes.nuscenes as nuscenes_module
import NuscenesDataset.nuscenes.utils.splits
import NuscenesDataset.nuscenes.utils.data_classes as dc


class DatasetBuilder:

    def __init__(self, args, map, isTrain=True):

        # basic settings
        exp_type = 'train' if isTrain else 'valid'

        # load nuscenes loader
        self.nusc = map.nusc
        self.map = map
        self.model_name = args.model_name

        self.nuscenes_config(args)
        self.split_train_val_test()

        # params
        config = read_config()

        self.dataset_path = os.path.join(args.dataset_path, exp_type)
        self.obs_len = int(args.past_horizon_seconds * args.target_sample_period)
        self.pred_len = int(args.future_horizon_seconds * args.target_sample_period)
        self.min_num_agents = args.min_num_agents
        self.target_sample_period = args.target_sample_period
        self.scene_accept_prob = args.scene_accept_prob

        self.val_ratio = args.val_ratio
        self.preprocess = args.preprocess_trajectory
        self.num_turn_scene_repeats = args.num_turn_scene_repeats

        self.min_past_horizon_seconds = args.min_past_horizon_seconds
        self.min_future_horizon_seconds = args.min_future_horizon_seconds

        self.category_filtering_method = args.category_filtering_method

        self.use_preprocessed_lidar = args.use_preprocessed_lidar
        self.num_lidar_sweeps = args.num_lidar_sweeps
        self.x_range = (args.x_range_min, args.x_range_max)
        self.y_range = (args.y_range_min, args.y_range_max)
        self.z_range = (args.z_range_min, args.z_range_max)
        self.map_size = args.map_size

        self.num_stop_agents = 0
        self.num_moving_agents = 0
        self.num_turn_agents = 0

        # checks existance of dataset file
        self.voxel_save_dir = config['Nuscenes']['voxel_dir']
        if (os.path.exists(self.voxel_save_dir)==False):
            os.mkdir(self.voxel_save_dir)


    def make_preprocessed_data(self, file_path, exp_type):

        def transform_box(box, ego_pose):
            '''
            transform to ego pose
            '''
            box.transform_to_pose(ego_pose)


        print('>> Making preprocessed data ..')

        # set seed
        np.random.seed(1)

        # defin exp_type
        sample_tokens = self.train_sample_tokens if (exp_type == 'train') else self.test_sample_tokens
        sample_tokens_rnd = copy.deepcopy(sample_tokens)
        random.shuffle(sample_tokens_rnd)

        cnt = 0
        scene_list_train, scene_list_val, scene_list_test = [], [], []
        for tok in tqdm(sample_tokens_rnd):

            if (np.random.rand(1) > self.scene_accept_prob):
                continue

            # current sample (current time)
            sample = self.nusc.get('sample', tok)
            sample_token = sample['token']
            location = self.nusc.get('log', self.nusc.get('scene', sample['scene_token'])['log_token'])['location']

            # corresponding lidar_sample_data
            lidar_sample_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])

            # sequential lidar_sample_data
            lidar_future_sample_data = self.traverse_linked_list(lidar_sample_data, 'sample_data', 'next')
            lidar_past_sample_data = self.traverse_linked_list(lidar_sample_data, 'sample_data', 'prev')
            lidar_all_sample_data = lidar_past_sample_data + [lidar_sample_data] + lidar_future_sample_data
            lidar_tokens = [_['token'] for _ in lidar_all_sample_data]

            # sequential Lidar timestamps relative to now.
            timestamp_now = lidar_sample_data['timestamp']
            lidar_timestamps = [_['timestamp'] for _ in lidar_all_sample_data]
            lidar_timestamps_relative = [(_ - timestamp_now) / 1e6 for _ in lidar_timestamps]

            # not enough observation length
            if lidar_timestamps_relative[0] > self.target_past_times[0]:
                continue

            # not enough future length
            if lidar_timestamps_relative[-1] < self.target_future_times[-1]:
                continue


            # # ------------------------------------
            # Ego vehicle trajectory
            # #-------------------------------------

            # World extrinsics + intrinsics of rear axle on driving car.
            ego_pose_now = self.get_ego_pose(lidar_sample_data['ego_pose_token'])
            ego_poses_all = [self.get_ego_pose(_['ego_pose_token']) for _ in lidar_all_sample_data]
            ego_boxes_all = [dc.EgoBox.from_pose(_) for _ in ego_poses_all]

            # Interpolate boxes at the target times (global coordinates).
            ego_boxes_past_interp, lidar_tokens_past = self.interpolate_boxes_to_times(ego_boxes_all,
                                                    lidar_timestamps_relative, lidar_tokens, self.target_past_times)
            ego_boxes_future_interp, lidar_tokens_future = self.interpolate_boxes_to_times(ego_boxes_all,
                                                    lidar_timestamps_relative, lidar_tokens, self.target_future_times)
            ego_boxes_interp = ego_boxes_past_interp + ego_boxes_future_interp

            lidar_now_token = lidar_tokens_past[-1]
            lidar_token_seq = lidar_tokens_past + lidar_tokens_future


            # The "now" box is the most recent past.
            # ego_box_now = ego_boxes_past_interp[-1]

            # Create agent dictionary for the current scene
            agent_dict = {}
            agent_dict['EGO'] = Agent(type='VEHICLE', track_id='EGO', obs_len=self.obs_len, pred_len=self.pred_len)

            for idx in range(len(ego_boxes_interp)):
                agent_dict['EGO'].trajectory_global_coord[idx, 0] = self.target_all_times[idx]
                agent_dict['EGO'].trajectory_global_coord[idx, 1] = ego_boxes_interp[idx].center[0]
                agent_dict['EGO'].trajectory_global_coord[idx, 2] = ego_boxes_interp[idx].center[1]
                agent_dict['EGO'].trajectory_global_coord[idx, 3] = ego_boxes_interp[idx].center[2]
                agent_dict['EGO'].yaw_global[idx, 0] = ego_boxes_interp[idx].get_yaw()

            # From global coord. system to SDV coord. system at now
            _ = [transform_box(_, ego_pose_now) for _ in ego_boxes_interp]

            for idx in range(len(ego_boxes_interp)):
                agent_dict['EGO'].trajectory[idx, 0] = self.target_all_times[idx]
                agent_dict['EGO'].trajectory[idx, 1] = ego_boxes_interp[idx].center[0]
                agent_dict['EGO'].trajectory[idx, 2] = ego_boxes_interp[idx].center[1]
                agent_dict['EGO'].trajectory[idx, 3] = ego_boxes_interp[idx].center[2]

            agent_dict['EGO'].wlh = ego_boxes_past_interp[-1].wlh
            agent_dict['EGO'].yaw = ego_boxes_past_interp[-1].get_yaw()




            # # ------------------------------------
            # Neighboring Agents
            # #-------------------------------------
            num_agents = 1
            ann_tokens = sample['anns']
            for ann_token in ann_tokens:
                ann = self.nusc.get('sample_annotation', ann_token)

                category = ann['category_name']
                if len(ann['attribute_tokens']):
                    attribute = self.nusc.get('attribute', ann['attribute_tokens'][0])['name']
                else:
                    continue

                if (self.check_valid_agent(category, attribute) == False):
                    continue


                # ------------------------
                # consider agents who have valid trajectory length
                # ------------------------

                # Keyframed annotations.
                anns_future = self.traverse_linked_list(ann, 'sample_annotation', 'next')
                anns_past = self.traverse_linked_list(ann, 'sample_annotation', 'prev')
                anns = anns_past + [ann] + anns_future

                # Timestamps
                ann_timestamps = [self.nusc.get('sample', _['sample_token'])['timestamp'] for _ in anns]
                ann_timestamps_relative = [(_ - timestamp_now) / 1e6 for _ in ann_timestamps]

                # checking trajectory length
                if ann_timestamps_relative[0] > -1*self.min_past_horizon_seconds:
                    # Not enough agent past annotations. Skipping
                    continue

                if ann_timestamps_relative[-1] < self.min_future_horizon_seconds:
                    # Not enough agent future annotations. Skipping
                    continue

                # define target time indices
                for i in range(self.target_past_times.size):
                    check = np.array(ann_timestamps_relative) < self.target_past_times[i]
                    if (np.count_nonzero(check)):
                        target_past_times = self.target_past_times[i:]
                        break

                for i in range(self.target_future_times.size-1, -1, -1):
                    check = np.array(ann_timestamps_relative) > self.target_future_times[i]
                    if (np.count_nonzero(check)):
                        target_future_times = self.target_future_times[:i+1]
                        break

                # Included in an agent dictionary
                agent = Agent(type=category, attribute=attribute, track_id=ann['token'], obs_len=self.obs_len, pred_len=self.pred_len)

                # ------------------------
                # extract trajectory, bbox, and yaw info
                # ------------------------
                ann_boxes = [self.nusc.get_box(_['token']) for _ in anns]

                # Interpolate boxes at the target times (global coordinates.)
                ann_boxes_past_interp, _ = self.interpolate_boxes_to_times(ann_boxes, ann_timestamps_relative, lidar_tokens, target_past_times)
                ann_boxes_future_interp, _ = self.interpolate_boxes_to_times(ann_boxes, ann_timestamps_relative, lidar_tokens, target_future_times)
                ann_boxes_interp = ann_boxes_past_interp + ann_boxes_future_interp
                ann_target_all_times = np.concatenate([target_past_times, target_future_times])

                sort_indices = np.searchsorted(self.target_all_times, ann_target_all_times)
                for _ in range(sort_indices.size):
                    idx = sort_indices[_]
                    agent.trajectory_global_coord[idx, 0] = ann_target_all_times[_]
                    agent.trajectory_global_coord[idx, 1] = ann_boxes_interp[_].center[0]
                    agent.trajectory_global_coord[idx, 2] = ann_boxes_interp[_].center[1]
                    agent.trajectory_global_coord[idx, 3] = ann_boxes_interp[_].center[2]
                    agent.yaw_global[idx, 0] = ann_boxes_interp[_].get_yaw()

                # Annotation boxes in ego_pose now or lidar_pose now coordinates.
                _ = [transform_box(_, ego_pose_now) for _ in ann_boxes_interp]

                for _ in range(sort_indices.size):
                    idx = sort_indices[_]
                    agent.trajectory[idx, 0] = ann_target_all_times[_]
                    agent.trajectory[idx, 1] = ann_boxes_interp[_].center[0]
                    agent.trajectory[idx, 2] = ann_boxes_interp[_].center[1]
                    agent.trajectory[idx, 3] = ann_boxes_interp[_].center[2]

                agent.wlh = ann_boxes_past_interp[-1].wlh
                agent.yaw = ann_boxes_past_interp[-1].get_yaw()


                # ------------------------
                # store in an agent dictionary
                # ------------------------
                agent_dict[ann['token']] = agent
                num_agents += 1


            if (num_agents < self.min_num_agents):
                continue

            # updating agents information
            for idx, key in enumerate(agent_dict):
                agent_dict[key].agent_id = idx
                agent_dict[key].heading_from_traj()
                agent_dict[key].calc_speed(self.target_sample_period)
                agent_dict[key].trajectory_curvature()
                agent_dict[key].update_status()
                if ('scratch' in self.model_name):
                    agent_dict[key].possible_lanes = self.map.find_possible_lanes(agent_dict[key], lidar_now_token, location)
                    if (len(agent_dict[key].possible_lanes) > 0):
                        agent_dict[key].possible_lanes = self.map.find_best_lane(agent_dict[key].possible_lanes,
                                                                            agent_dict[key].trajectory_global_coord[:, 1:],
                                                                            location,
                                                                            self.obs_len)

            # debug ------
            # ego_timestamps = agent_dict['EGO'].trajectory[:, 0]
            # for idx, key in enumerate(agent_dict):
            #     agent = agent_dict[key]
            #     agent_timestamps = agent.trajectory[:, 0]
            #
            #     diff = ego_timestamps - agent_timestamps
            #     if (np.sum(np.abs(diff[~np.isnan(diff)])) > 0):
            #         print("false interpolation")
            #  debug ------

            # # ----------------------------
            # Voxel pre-calculation
            # # ----------------------------
            if (self.use_preprocessed_lidar == 1):
                lidar_past_sample_data = self.traverse_linked_list(lidar_sample_data, 'sample_data', 'prev') + [lidar_sample_data]
                past_lidar_sweeps_tokens = [_['token'] for _ in lidar_past_sample_data]
                voxels = self.map.make_topview_voxels(past_lidar_sweeps_tokens[-self.num_lidar_sweeps:], self.x_range, self.y_range, self.z_range, self.map_size)

                voxel_dict = {}
                voxel_dict['x_min'] = self.x_range[0]
                voxel_dict['x_max'] = self.x_range[1]
                voxel_dict['y_min'] = self.y_range[0]
                voxel_dict['y_max'] = self.y_range[1]
                voxel_dict['z_min'] = self.z_range[0]
                voxel_dict['z_max'] = self.z_range[1]
                voxel_dict['num_lidar_sweeps'] = self.num_lidar_sweeps
                voxel_dict['voxels'] = voxels.astype('uint8')

                file_name = sample_token + '.pkl'
                f = open(os.path.join(self.voxel_save_dir, file_name), 'wb')
                pickle.dump(voxel_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.close()

            # save in the scene
            scene = Scene(sample_token=sample_token, lidar_token_seq=lidar_token_seq, agent_dict=agent_dict, city_name=location)
            scene.make_id_2_token_lookup()

            # Split into train/valid/test ------------------
            if (exp_type == 'train'):
                if (np.random.rand(1) < self.val_ratio):
                    scene_list_val.append(scene)
                else:
                    scene_list_train.append(scene)
            else:
                scene_list_test.append(scene)

            cnt += 1


        with open(file_path, 'wb') as f:
            dill.dump([scene_list_train, scene_list_val, scene_list_test,
                       self.num_stop_agents, self.num_moving_agents, self.num_turn_agents], f, protocol=dill.HIGHEST_PROTOCOL)
        print('>> {%s} is created .. ' % file_path)

    def preprocess_lidar_sweeps(self, data):

        num_scenes = len(data)
        for _, scene in enumerate(data):

            # current sample (current time)
            sample_token = scene.sample_token
            sample = self.nusc.get('sample', sample_token)
            file_name = sample_token + '.pkl'
            if (os.path.exists(os.path.join(self.voxel_save_dir, file_name))==False):

                # corresponding lidar_sample_data
                lidar_sample_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])

                lidar_past_sample_data = self.traverse_linked_list(lidar_sample_data, 'sample_data', 'prev') + [lidar_sample_data]
                past_lidar_sweeps_tokens = [_['token'] for _ in lidar_past_sample_data]
                voxels = self.map.make_topview_voxels(past_lidar_sweeps_tokens[-self.num_lidar_sweeps:], self.x_range,
                                                      self.y_range, self.z_range, self.map_size)

                voxel_dict = {}
                voxel_dict['x_min'] = self.x_range[0]
                voxel_dict['x_max'] = self.x_range[1]
                voxel_dict['y_min'] = self.y_range[0]
                voxel_dict['y_max'] = self.y_range[1]
                voxel_dict['z_min'] = self.z_range[0]
                voxel_dict['z_max'] = self.z_range[1]
                voxel_dict['num_lidar_sweeps'] = self.num_lidar_sweeps
                voxel_dict['voxels'] = voxels.astype('uint8')


                f = open(os.path.join(self.voxel_save_dir, file_name), 'wb')
                pickle.dump(voxel_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.close()

            print_voxelization_progress(_, num_scenes)

    def nuscenes_config(self, args):

        '''
        Samples are at 2Hz
        LIDAR is at 20Hz
        '''

        '''
        target vehicle category and attribute are based on official prediction benchmark
        (https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/tutorials/prediction_tutorial.ipynb)
        '''

        self.target_vehicle_category = ['vehicle.car', 'vehicle.bus.rigid', 'vehicle.truck', 'vehicle.bus.bendy',
                                        'vehicle.emergency.police', 'vehicle.construction']
        self.target_vehicle_attribute = ['vehicle.moving', 'vehicle.stopped', 'vehicle.parked']

        # Predict future_horizon_seconds in the future with past_horizon_seconds.
        past_horizon_seconds = args.past_horizon_seconds
        future_horizon_seconds = args.future_horizon_seconds

        # Configuration for temporal interpolation.
        target_sample_period_past = args.target_sample_period # Hz
        target_sample_period_future = args.target_sample_period
        target_sample_frequency_past = 1. / target_sample_period_past # sec
        target_sample_frequency_future = 1. / target_sample_period_future

        self.target_past_times = -1 * np.arange(0, past_horizon_seconds, target_sample_frequency_past)[::-1]
        self.target_past_times[np.isclose(self.target_past_times, 0.0, atol=1e-5)] = 1e-8
        self.target_future_times = np.arange(target_sample_frequency_future, future_horizon_seconds
                                             + target_sample_frequency_future, target_sample_frequency_future)

        self.target_all_times = np.concatenate([self.target_past_times, self.target_future_times])

    def traverse_linked_list(self, obj, tablekey, direction, inclusive=False):
        return nuscenes_module.traverse_linked_list(self.nusc, obj, tablekey, direction, inclusive)

    def interpolate_boxes_to_times(self, boxes, box_timestamps_relative, lidar_tokens, relative_times_interp):
        return nuscenes_module.interpolate_boxes_to_times(boxes, box_timestamps_relative, lidar_tokens, relative_times_interp)

    def scene_to_sample_tokens(self, scene):
        samples = nuscenes_module.traverse_linked_list(self.nusc, self.nusc.get('sample', scene['first_sample_token']),
                                                       'sample', 'next', inclusive=True)
        return [sample['token'] for sample in samples]

    def split_train_val_test(self):

        '''
        Split scenes into train/valid/test accroding to Trajectron++ (ECCV, 2020)
        '''

        scene_blacklist = [499, 515, 517]

        # split scene names
        train_scenes = NuscenesDataset.nuscenes.utils.splits.train
        # val_scenes = []
        test_scenes = NuscenesDataset.nuscenes.utils.splits.val

        # assign ID to each scene
        train_scene_inds = [self.nusc.scene_name_to_scene_idx[_] for _ in train_scenes]
        # val_scene_inds = []
        test_scene_inds = [self.nusc.scene_name_to_scene_idx[_] for _ in test_scenes]

        # remove scenes in black_list
        for _, id in enumerate(scene_blacklist):
            if (id in train_scene_inds):
                train_scene_inds.remove(id)

            # if (id in val_scene_inds):
            #     val_scene_inds.remove(id)

            if (id in test_scene_inds):
                test_scene_inds.remove(id)

        # all the sample tokens related to train/val/test scenes
        self.train_sample_tokens = []
        self.val_sample_tokens = []
        self.test_sample_tokens = []

        for ti in train_scene_inds:
            self.train_sample_tokens.extend(self.scene_to_sample_tokens(self.nusc.scene[ti]))

        # for vi in val_scene_inds:
        #     self.val_sample_tokens.extend(self.scene_to_sample_tokens(self.nusc.scene[vi]))

        for tei in test_scene_inds:
            self.test_sample_tokens.extend(self.scene_to_sample_tokens(self.nusc.scene[tei]))

    def get_ego_pose(self, pose_token):
        return self.nusc.get('ego_pose', pose_token)

    def check_valid_agent(self, category, attribute):


        if (self.category_filtering_method == 0):
            # ------------------------
            # According to Trajectron++ (ECCV, 2020)
            # ------------------------
            if 'vehicle' in category and 'bicycle' not in category and 'motorcycle' not in category and 'parked' not in attribute:
                return True
            else:
                return False

        elif (self.category_filtering_method == 1):
            # ------------------------
            # According to Nuscenes prediction benchmark
            # ------------------------
            if (category in self.target_vehicle_category and attribute in self.target_vehicle_attribute):
                return True
            else:
                return False

        elif (self.category_filtering_method == 2):
            # ------------------------
            # According to Trajectron++ (ECCV, 2020)
            # ------------------------
            ped_flag = False
            if 'pedestrian' in category and not 'stroller' in category and not 'wheelchair' in category:
                ped_flag = True

            veh_flag = False
            if 'vehicle' in category and 'bicycle' not in category and 'motorcycle' not in category and 'parked' not in attribute:
                veh_flag = True

            return (ped_flag or veh_flag)

        else:
            print("[Error] method %d is not supported !!" % self.category_filtering_method)
            sys.exit(0)




def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', type=str, default='/home/dooseop/DATASET/nuscenes/')
    parser.add_argument('--version', type=str, default='v1.0-trainval')

    parser.add_argument('--past_horizon_seconds', type=float, default=2)
    parser.add_argument('--future_horizon_seconds', type=float, default=4)
    parser.add_argument('--target_sample_period', type=float, default=5)

    parser.add_argument('--min_num_agents', type=int, default=1)
    parser.add_argument('--obs_len', type=int, default=10)
    parser.add_argument('--pred_len', type=int, default=20)

    parser.add_argument('--preprocess', type=int, default=0)
    parser.add_argument('--num_turn_scene_repeats', type=int, default=0)
    parser.add_argument('--val_ratio', type=float, default=0.05)


    args = parser.parse_args()

    builder = DatasetBuilder(args)

    for i in range(32, 100):
        a = 0
        # print(">> scene_number : %d, angle: %.2f, location: %s" % (i, angle-90, location))

if __name__ == '__main__':
    main()