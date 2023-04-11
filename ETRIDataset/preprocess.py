from utils.functions import read_config, print_voxelization_progress
from utils.libraries import *
from ETRIDataset.agent import Agent
from ETRIDataset.scene import Scene
from ETRIDataset.VossHelper import VossHelper, Pose

class DatasetBuilder:

    def __init__(self, args, map, isTrain=True):


        # basic settings
        exp_type = 'train' if isTrain else 'test'

        self.etri_config(args)
        self.split_train_val_test()

        # params
        config = read_config()

        self.dataset_path = args.dataset_path
        self.obs_len = int(args.past_horizon_seconds * args.target_sample_period)
        self.pred_len = int(args.future_horizon_seconds * args.target_sample_period)
        self.sub_step = int(10.0 / args.target_sample_period)
        self.min_num_agents = args.min_num_agents
        self.target_sample_period = args.target_sample_period
        self.scene_accept_prob = args.scene_accept_prob

        self.val_ratio = args.val_ratio
        self.preprocess = args.preprocess_trajectory
        self.num_turn_scene_repeats = args.num_turn_scene_repeats

        self.min_past_horizon_seconds = args.min_past_horizon_seconds
        self.min_future_horizon_seconds = args.min_future_horizon_seconds

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
        self.voxel_save_dir = config['ETRI']['voxel_dir']
        if (os.path.exists(self.voxel_save_dir)==False):
            os.mkdir(self.voxel_save_dir)

        # voss helper
        self.vh = VossHelper()


    def make_preprocessed_data(self, file_path, exp_type):

        print('>> Making preprocessed data ..')

        # set seed
        np.random.seed(1)

        if (exp_type == 'train'):
            target_scenes = self.train_scenes
        else:
            target_scenes = self.test_scenes

        scene_list_train, scene_list_val, scene_list_test = [], [], []
        for scene_name in tqdm(target_scenes):
            target_scene_path = os.path.join(self.dataset_path, scene_name)

            '''
            ------------------------------------------------------------------------------------------------
            frame index| class | obj_id | x[m] | y[m] |  z[m]  | heading[rad] | width[m] | length[m] | height[m] | lidar | image
            ------------------------------------------------------------------------------------------------
            0          | 1     | 2      | 3    | 4    |   5    |      6       |    7     | 8         | 9         | 10     | 11
            ------------------------------------------------------------------------------------------------
            '''
            data = self.vh.read_raw_file(os.path.join(target_scene_path, 'label_ori.csv'))[1:]
            data_proc = self.vh.read_raw_file(os.path.join(target_scene_path, 'label_proc.csv'))[1:]
            frm_indices = np.unique(data[:, 0]).astype('int')
            num_scenes = len(frm_indices)

            # for all time index in the log -------------------
            pivot = 15
            ref_time_index_start = (self.sub_step * self.obs_len)-1 + pivot
            ref_time_index_end = num_scenes - (self.sub_step * self.pred_len) - 1
            for curr_time_index in range(ref_time_index_start, ref_time_index_end, 1):

                # target duration
                tar_frm_index_start = frm_indices[curr_time_index] - (self.sub_step * self.obs_len) + 1
                tar_frm_index_end = frm_indices[curr_time_index] + (self.sub_step * self.pred_len)

                # current pose of ego-vehicle
                ref_frm_index = [t for t in range(tar_frm_index_start, tar_frm_index_end+1, self.sub_step)][self.obs_len - 1]

                # if (ref_frm_index == 54):
                #     break_point = 0

                ref_pose = self.vh.get_pose(data, ref_frm_index, -1)
                lidar_file_name, image_file_name = self.vh.get_sensor_file_names(data, ref_frm_index)

                # Create agent dictionary for the current scene
                agent_dict = {}
                agent_dict['EGO'] = Agent(type='vehicle', track_id='EGO', obs_len=self.obs_len, pred_len=self.pred_len) # update, 220106

                # gather basic info (class, id, seq_lens) -------------------
                for tar_frm_index in range(tar_frm_index_start, tar_frm_index_end + 1, self.sub_step):
                    objects = self.vh.get_label_object(data, tar_frm_index)
                    for obj in objects:
                        if (obj['obj_id'] not in agent_dict and obj['obj_class'] not in self.exceptions):
                            agent = Agent(type=obj['obj_class'], track_id=obj['obj_id'], obs_len=self.obs_len, pred_len=self.pred_len)
                            agent_dict[obj['obj_id']] = agent

                # Calculate trajectories -------------------
                cnt = 0
                for tar_frm_index in range(tar_frm_index_start, tar_frm_index_end + 1, self.sub_step):

                    # ego-vehicle
                    tgt_pose = self.vh.get_pose(data, tar_frm_index, -1)
                    x_e, y_e = ref_pose.to_agent(tgt_pose.position)[0]
                    agent_dict['EGO'].trajectory[cnt, 0] = tar_frm_index
                    agent_dict['EGO'].trajectory[cnt, 1] = x_e
                    agent_dict['EGO'].trajectory[cnt, 2] = y_e
                    agent_dict['EGO'].trajectory[cnt, 3] = tgt_pose.xyz[0, 2]


                    if (tar_frm_index == ref_frm_index):
                        agent_dict['EGO'].pose = tgt_pose

                    # other agents
                    objects = self.vh.get_label_object(data, tar_frm_index)
                    for obj in objects:

                        if (obj['obj_id'] not in agent_dict):
                            continue

                        # store data
                        obj_pose = self.vh.get_pose(data, tar_frm_index, obj['obj_id'])
                        x_o, y_o = ref_pose.to_agent(obj_pose.position)[0]
                        agent_dict[obj['obj_id']].trajectory[cnt, 0] = tar_frm_index
                        agent_dict[obj['obj_id']].trajectory[cnt, 1] = x_o
                        agent_dict[obj['obj_id']].trajectory[cnt, 2] = y_o
                        agent_dict[obj['obj_id']].trajectory[cnt, 3] = obj_pose.xyz[0, 2]

                        try:
                            obj_pose_proc = self.vh.get_pose(data_proc, tar_frm_index, obj['obj_id'])
                            x_o, y_o = ref_pose.to_agent(obj_pose_proc.position)[0]
                            z_o = obj_pose_proc.xyz[0, 2]
                        except:
                            x_o, y_o, z_o = np.nan, np.nan, np.nan

                        agent_dict[obj['obj_id']].trajectory_proc[cnt, 0] = tar_frm_index
                        agent_dict[obj['obj_id']].trajectory_proc[cnt, 1] = x_o
                        agent_dict[obj['obj_id']].trajectory_proc[cnt, 2] = y_o
                        agent_dict[obj['obj_id']].trajectory_proc[cnt, 3] = z_o

                        if (tar_frm_index == ref_frm_index):
                            agent_dict[obj['obj_id']].pose = obj_pose
                            agent_dict[obj['obj_id']].wlh = obj['wlh']

                    cnt+=1

                # # Calculate additional info -------------------
                for idx, key in enumerate(agent_dict):
                    agent_dict[key].agent_id = idx
                    if (agent_dict[key].wlh is not None):
                        agent_dict[key].bbox2D()
                        agent_dict[key].bbox_e = ref_pose.to_agent(agent_dict[key].bbox_g)

                # save in the scene
                scene = Scene(log_token=scene_name, time_index=ref_frm_index, agent_dict=agent_dict)
                scene.make_id_2_token_lookup()
                scene.lidar_file_name = lidar_file_name
                scene.image_file_name = image_file_name

                # Split into train/valid/test ------------------
                if (exp_type == 'train'):
                    if (np.random.rand(1) < self.val_ratio):
                        scene_list_val.append(scene)
                    else:
                        scene_list_train.append(scene)
                else:
                    scene_list_test.append(scene)

        with open(file_path, 'wb') as f:
            dill.dump([scene_list_train, scene_list_val, scene_list_test,
                       self.num_stop_agents, self.num_moving_agents, self.num_turn_agents], f, protocol=dill.HIGHEST_PROTOCOL)
        print('>> {%s} is created .. ' % file_path)


    def etri_config(self, args):

        '''
        Samples are at 10Hz
        LIDAR is at 10Hz
        '''

        self.exceptions = ['unknown'] # update, 220106


    def split_train_val_test(self):

        R1 = ['0001', '0020', '0075', '0076', '0077', '0143', '0144', '0145']
        R2 = ['0002', '0003', '0004', '0043', '0044', '0045', '0046', '0047', '0048', '0072', '0073', '0074', '0146']
        R3 = ['0071', '0100', '0106', '0107', '0140', '0141', '0142']
        R4 = ['0006', '0007', '0008', '0036', '0037', '0038', '0039', '0069', '0070']
        R6 = ['0009', '0010', '0035', '0068']
        R7 = ['0011', '0066', '0067']
        R8 = ['0012', '0013', '0033', '0034', '0064', '0138']
        R9 = ['0014', '0016', '0032', '0063']
        R10 = ['0015', '0017', '0018', '0019']
        R11 = ['0031', '0060', '0061', '0062', '0119', '0135', '0136', '0137']
        R12 = ['0029', '0030', '0056', '0057', '0058']
        R15 = ['0050']
        R16 = ['0109', '0110', '0111', '0123', '0125', '0126']

        R1.remove('0075')
        R2.remove('0043')
        R3.remove('0140')
        R4.remove('0006')
        R8.remove('0033')
        R9.remove('0016')
        R11.remove('0135')
        R16.remove('0110')

        self.train_scenes = R1 + R2 + R3 + R4 + R6 + R7 + R8 + R9 + R10 + R11 + R12 + R15 + R16
        self.test_scenes = ['0075', '0043', '0140', '0006', '0033', '0016', '0135', '0110']


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

