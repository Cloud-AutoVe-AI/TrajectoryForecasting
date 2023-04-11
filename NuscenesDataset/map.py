from utils.functions import *
from shapely import affinity
import colorsys
from shapely.geometry import MultiPolygon
from NuscenesDataset.nuscenes.map_expansion.arcline_path_utils import discretize_lane
from NuscenesDataset.nuscenes.utils.geometry_utils import transform_matrix
from NuscenesDataset.nuscenes.map_expansion.map_api import NuScenesMap

import NuscenesDataset.nuscenes.utils.data_classes as dc

class Map:

    def __init__(self, args, nusc):

        self.nusc = nusc

        # Nuscenes Map loader
        self.nusc_maps = {}
        self.nusc_maps['singapore-onenorth'] = NuScenesMap(dataroot=args.dataset_path, map_name='singapore-onenorth')
        self.nusc_maps['singapore-hollandvillage'] = NuScenesMap(dataroot=args.dataset_path, map_name='singapore-hollandvillage')
        self.nusc_maps['singapore-queenstown'] = NuScenesMap(dataroot=args.dataset_path, map_name='singapore-queenstown')
        self.nusc_maps['boston-seaport'] = NuScenesMap(dataroot=args.dataset_path, map_name='boston-seaport')

        self.centerlines = {}
        self.centerlines['singapore-onenorth'] = self.nusc_maps['singapore-onenorth'].discretize_centerlines(resolution_meters=0.25)
        self.centerlines['singapore-hollandvillage'] = self.nusc_maps['singapore-hollandvillage'].discretize_centerlines(resolution_meters=0.25)
        self.centerlines['singapore-queenstown'] = self.nusc_maps['singapore-queenstown'].discretize_centerlines(resolution_meters=0.25)
        self.centerlines['boston-seaport'] = self.nusc_maps['boston-seaport'].discretize_centerlines(resolution_meters=0.25)

        # params
        self.map_size = args.map_size
        self.lidar_map_ch_dim = args.lidar_map_ch_dim
        self.num_lidar_sweeps = args.num_lidar_sweeps
        self.min_forward_len = 80
        self.min_backward_len = 10
        self.lidar_sample_period = 20 # Hz

        # control params
        self.centerline_width = args.centerline_width
        try:
            self.is_draw_centerlines = args.is_draw_centerlines
        except:
            self.is_draw_centerlines = 1
        try:
            self.hdmap_type = args.hdmap_type
        except:
            self.hdmap_type = 0


    # ------------------------------------------
    # ETC
    # ------------------------------------------

    def load_point_cloud_file(self, token):
        lidar_sample_data = self.nusc.get('sample_data', token)
        pcl_path = os.path.join(self.nusc.dataroot, lidar_sample_data['filename'])
        pc = dc.LidarPointCloud.from_file(pcl_path).points[:3]
        return pc

    def transform_pc(self, R, pc):
        pc = np.matmul(R, np.concatenate([pc, np.ones(shape=(1, pc.shape[1]))], axis=0))[:3]
        return pc

    def _transform_pc(self, R, translation, pc):
        pc_trans = pc - translation.reshape(1, 3)
        return np.matmul(R, pc_trans.T).T

    def _transform_pc_inv(self, R, translation, pc):
        pc_trans_inv = np.matmul(R, pc.T).T + translation.reshape(1, 3)
        return pc_trans_inv


    # ------------------------------------------
    # Voxelization
    # ------------------------------------------

    def pc_voxelization(self, PCs, x_range, y_range, z_range, map_size):

        '''
        The center of the point cloud is shifted and translated to match
        the center of the ego-vehicle

        pc[:, 0] : x-axis, forward
        pc[:, 1] : y-axis, left
        pc[:, 2] : z-axis, up
        '''

        def topview_transform(img, pc, x_range, y_range, z_range, map_size):

            x = pc[0, :]
            y = pc[1, :]
            z = pc[2, :]

            axis_range_y = y_range[1] - y_range[0]
            axis_range_x = x_range[1] - x_range[0]
            scale_y = float(map_size - 1) / axis_range_y
            scale_x = float(map_size - 1) / axis_range_x

            # extract in-range points
            x_lim = in_range_points(x, x, y, z, x_range, y_range, z_range)
            y_lim = in_range_points(y, x, y, z, x_range, y_range, z_range)

            # to pixel domain
            col_img = -(y_lim * scale_y).astype(np.int32)
            row_img = -(x_lim * scale_x).astype(np.int32)

            col_img += int(np.trunc(y_range[1] * scale_y))
            row_img += int(np.trunc(x_range[1] * scale_x))

            img[row_img.astype('int32'), col_img.astype('int32')] += np.ones_like(x_lim)

            # cv2.imshow("", (255*img).astype('uint8'))
            # cv2.waitKey(0)

            return img

        def voxcelization(pc, num_grid):

            img = np.zeros(shape=(map_size, map_size, int(num_grid)))
            z_min_w, z_max_w = z_range
            grid_size = (z_max_w - z_min_w) / float(num_grid)
            for g in range(int(num_grid)):
                z_min = g * grid_size + z_min_w
                z_max = grid_size + z_min
                img[:, :, g] = topview_transform(img[:, :, g], pc, x_range, y_range, (z_min, z_max), map_size)

            return img

        # params
        num_grid = self.lidar_map_ch_dim
        nsweeps = len(PCs)

        voxel = voxcelization(PCs[0], num_grid)
        for i in range(1, nsweeps):
            voxel = np.concatenate([voxel, voxcelization(PCs[i], num_grid)], axis=2)

        '''
        voxel = [voxel(t) voxel(t-1) voxel(t-2)...]
        '''
        return voxel

    def make_topview_voxels(self, token_seq, x_range, y_range, z_range, map_size):

        '''
        " ... a bounding box defining the position of an object seen in a sample. All location
        data is given with respect to the global coordinate system." (https://www.nuscenes.org/nuscenes#data-formate)

        token_seq : list of consecutive lidar tokens
        * The last element is reference
        '''

        nsweeps = len(token_seq)

        # Reference sample data
        ref_sd = self.nusc.get('sample_data', token_seq[-1])
        ref_pose = self.nusc.get('ego_pose', ref_sd['ego_pose_token'])

        # From global to _current_ ego car frame
        car_from_global = transform_matrix(ref_pose['translation'], pyquaternion.Quaternion(ref_pose['rotation']), inverse=True)

        PCs = []
        # Imgs = []
        for _ in range(nsweeps):

            # Current sample data
            current_sd = self.nusc.get('sample_data', token_seq[(nsweeps-1)-_])

            # Load up the pointcloud.
            pc = self.load_point_cloud_file(current_sd['token'])


            # Get past pose.
            current_pose = self.nusc.get('ego_pose', current_sd['ego_pose_token'])
            global_from_car = transform_matrix(current_pose['translation'], pyquaternion.Quaternion(current_pose['rotation']), inverse=False)

            # From sensor coordinate frame to ego car frame.
            current_cs = self.nusc.get('calibrated_sensor', current_sd['calibrated_sensor_token'])
            car_from_current = transform_matrix(current_cs['translation'], pyquaternion.Quaternion(current_cs['rotation']), inverse=False)

            pc = self.transform_pc(car_from_current, pc)
            pc = self.transform_pc(global_from_car, pc)
            pc = self.transform_pc(car_from_global, pc)

            # debug ---
            # x = pc[0, :]
            # y = pc[1, :]
            # z = pc[2, :]
            #
            # x_lim = in_range_points(x, x, y, z, (-70, 70), (-70, 70), (-1, 3))
            # y_lim = in_range_points(y, x, y, z, (-70, 70), (-70, 70), (-1, 3))
            # z_lim = in_range_points(z, x, y, z, (-70, 70), (-70, 70), (-1, 3))
            #
            # len_ = x_lim.size
            #
            # pc_lim = np.concatenate([x_lim.reshape(len_, 1), y_lim.reshape(len_, 1), z_lim.reshape(len_, 1)], axis=1)
            #
            # import open3d
            # pcd = open3d.geometry.PointCloud()
            # pcd.points = open3d.utility.Vector3dVector(pc_lim)
            # open3d.visualization.draw_geometries([pcd])
            # debug ---


            PCs.append(pc)

            # img = self.draw_point_cloud_topview(pc, (-50, 50), (-50, 50), (-10, 10), map_size=500)
            # Imgs.append(img[:, :, 2].reshape(500, 500, 1))

        # topview = np.mean(np.concatenate(Imgs, axis=2), axis=2).reshape(500, 500, 1)

        return self.pc_voxelization(PCs, x_range, y_range, z_range, map_size)

    # ------------------------------------------
    # Draw HD map
    # ------------------------------------------

    def transfrom_matrix(self, R, T, inverse):

        tm = np.eye(4)

        if inverse:
            rot_inv = R
            trans = np.transpose(-np.array(T))
            tm[:3, :3] = rot_inv
            tm[:3, 3] = rot_inv.dot(trans)
        else:
            tm[:3, :3] = R
            tm[:3, 3] = np.transpose(np.array(T))

        return tm

    def lane_tok_to_coord(self, lanes, R, scene_location):

        coord_list = []
        for _, tok in enumerate(lanes):

            # get lane record
            lane_record = self.nusc_maps[scene_location].get_arcline_path(tok)

            # get coordinates
            coords = np.array(discretize_lane(lane_record, resolution_meters=0.25))
            coord_list.append(self.transform_pc(R, coords.T).T)

        return coord_list

    def make_topview_map(self, ego_pose, scene_location, x_range, y_range, map_size):

        '''
        " drivable surface polygons, road polygons, intersection polygons, vehicle lane polygons going straight
        dedicated left and right vehicle lane polygons, dedicated bike lane polygons, dedicated bus lane polygons,
        centerline markers for all lanes, lane dividers for all lanes with semantics ... This gives us a total of
        13 different map channels combining these elements." ILVM ECCV 2020.

        '''

        # ego-pose
        xyz = ego_pose['translation']
        R = transform_matrix(ego_pose['translation'], pyquaternion.Quaternion(ego_pose['rotation']), inverse=False)
        Rinv = transform_matrix(ego_pose['translation'], pyquaternion.Quaternion(ego_pose['rotation']), inverse=True)
        v = np.dot(R[:3, :3], np.array([1, 0, 0]))
        yaw = np.arctan2(v[1], v[0])

        static_scenes = []

        # centerlines (map_size x map_size x 1)
        # https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/tutorials/map_expansion_tutorial.ipynb
        if (self.is_draw_centerlines == 1):
            img_centerlines = np.zeros(shape=(map_size, map_size, 1)).astype('uint8')
            img_centerlines = self.draw_centerlines(img_centerlines, Rinv, xyz, x_range, y_range, map_size, scene_location).astype('float')
            static_scenes.append(img_centerlines)

        # cv2.imshow("", (255*img_centerlines).astype('uint8'))
        # cv2.waitKey(0)

        # drivable space (map_size x map_size x 1)
        img_drivable = self.draw_drivable_space(xyz, yaw, x_range, y_range, map_size, scene_location)
        static_scenes.append(img_drivable)

        # cv2.imshow("", (255*img_drivable).astype('uint8'))
        # cv2.waitKey(0)

        # road segment (map_size x map_size x 2)
        img_roadseg = self.draw_road_segment(xyz, yaw, x_range, y_range, map_size, scene_location)
        static_scenes.append(img_roadseg)


        # cv2.imshow("", (255*img_roadseg[:, :, 0]).astype('uint8'))
        # cv2.waitKey(0)

        # others (map_size x map_size x 1)
        img_pedcross = self.draw_others(xyz, yaw, x_range, y_range, map_size, scene_location, 'ped_crossing')
        img_walkway = self.draw_others(xyz, yaw, x_range, y_range, map_size, scene_location, 'walkway')
        img_trafficlight = self.draw_others(xyz, yaw, x_range, y_range, map_size, scene_location, 'stop_line')
        static_scenes.append(img_pedcross)
        static_scenes.append(img_walkway)
        static_scenes.append(img_trafficlight)

        output = np.concatenate(static_scenes, axis=2)
        output[output == 0] = -1.0

        return output

    def make_topview_map_agent_centric(self, possible_lanes, xyz, R, Rinv, scene_location, x_range, y_range, map_size):

        '''
        " drivable surface polygons, road polygons, intersection polygons, vehicle lane polygons going straight
        dedicated left and right vehicle lane polygons, dedicated bike lane polygons, dedicated bus lane polygons,
        centerline markers for all lanes, lane dividers for all lanes with semantics ... This gives us a total of
        13 different map channels combining these elements." ILVM ECCV 2020.

        '''

        R = self.transfrom_matrix(R, xyz, inverse=False)
        Rinv = self.transfrom_matrix(Rinv, xyz, inverse=True)
        v = np.dot(R[:3, :3], np.array([1, 0, 0]))
        yaw = np.arctan2(v[1], v[0])

        lane_coord_list = []
        num_lanes = len(possible_lanes)
        for i in range(num_lanes):
            lane_coord_list.append(self.lane_tok_to_coord(possible_lanes[i], Rinv, scene_location))

        static_scenes = []

        # centerlines (map_size x map_size x 3)
        # https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/tutorials/map_expansion_tutorial.ipynb
        if (self.is_draw_centerlines == 1):
            img_centerlines = np.zeros(shape=(map_size, map_size, 3)).astype('uint8')
            img_centerlines = self.draw_target_centerlines(img_centerlines, lane_coord_list, x_range, y_range, map_size).astype('float')
            static_scenes.append(img_centerlines)


        # cv2.imshow("", (255*img_centerlines).astype('uint8'))
        # cv2.waitKey(0)

        # drivable space (map_size x map_size x 1)
        img_drivable = self.draw_drivable_space(xyz, yaw, x_range, y_range, map_size, scene_location)
        static_scenes.append(img_drivable)


        # cv2.imshow("", (255*img_drivable).astype('uint8'))
        # cv2.waitKey(0)

        # road segment (map_size x map_size x 2)
        img_roadseg = self.draw_road_segment(xyz, yaw, x_range, y_range, map_size, scene_location)
        static_scenes.append(img_roadseg)

        # cv2.imshow("", (255*img_roadseg[:, :, 0]).astype('uint8'))
        # cv2.waitKey(0)

        # others (map_size x map_size x 1)
        img_pedcross = self.draw_others(xyz, yaw, x_range, y_range, map_size, scene_location, 'ped_crossing')
        img_walkway = self.draw_others(xyz, yaw, x_range, y_range, map_size, scene_location, 'walkway')
        img_trafficlight = self.draw_others(xyz, yaw, x_range, y_range, map_size, scene_location, 'stop_line')
        static_scenes.append(img_pedcross)
        static_scenes.append(img_walkway)
        static_scenes.append(img_trafficlight)

        output = np.concatenate(static_scenes, axis=2)
        output[output == 0] = -1.0

        return output

    def make_topview_map_loadertypeD(self, ego_pose, scene_location, x_range, y_range, map_size, obs_traj, category):

        # ego-pose
        xyz = ego_pose['translation']
        R = transform_matrix(ego_pose['translation'], pyquaternion.Quaternion(ego_pose['rotation']), inverse=False)
        Rinv = transform_matrix(ego_pose['translation'], pyquaternion.Quaternion(ego_pose['rotation']), inverse=True)
        v = np.dot(R[:3, :3], np.array([1, 0, 0]))
        yaw = np.arctan2(v[1], v[0])

        if (self.hdmap_type == 0):
            # map_size x map_size x 1, 0~255 (uint8)
            img_pedcross = (64.0 * self.draw_others(xyz, yaw, x_range, y_range, map_size, scene_location,
                                                    'ped_crossing')).astype('uint8')

            # map_size x map_size x 3, 0~255 (float)
            img_centerlines = np.concatenate([img_pedcross, img_pedcross, img_pedcross], axis=2)
            img_centerlines = self.draw_centerlines(img_centerlines, Rinv, xyz, x_range, y_range, map_size,
                                                    scene_location).astype('float')

            # map_size x map_size x 1, 0~255 (float)
            img_traj = self.draw_agent_trajectories(obs_traj, x_range, y_range, map_size, category)[:, :, 0].reshape(
                map_size, map_size, 1)

            # cv2.imshow("", img_centerlines.astype('uint8'))
            # cv2.waitKey(0)

            # map_size x map_size x 4, 0~255 (float)
            img = np.concatenate([img_centerlines, img_traj], axis=2)

        else:

            # map_size x map_size x 1, 0~255 (float)
            img_pedcross = 255.0*self.draw_others(xyz, yaw, x_range, y_range, map_size, scene_location, 'ped_crossing').reshape(map_size, map_size, 1)

            # map_size x map_size x 4, 0~255 (float)
            img_centerlines = self.draw_centerlines_multich(Rinv, xyz, x_range, y_range, map_size, scene_location).astype('float')

            # map_size x map_size x 1, 0~255 (float)
            img_traj = self.draw_agent_trajectories(obs_traj, x_range, y_range, map_size, category)[:, :, 0].reshape(map_size, map_size, 1)

            # map_size x map_size x 6, 0~255 (float)
            img = np.concatenate([img_pedcross, img_centerlines, img_traj], axis=2)

        # img = np.sum(img, axis=2).reshape(map_size, map_size, 1)
        # cv2.imshow("", img.astype('uint8'))
        # cv2.waitKey(0)

        return img.astype('float')/255.0

    def make_topview_map_loadertypeE(self, ego_pose, scene_location, x_range, y_range, map_size, obs_traj, category):


        # ego-pose
        xyz = ego_pose['translation']
        R = transform_matrix(ego_pose['translation'], pyquaternion.Quaternion(ego_pose['rotation']), inverse=False)
        Rinv = transform_matrix(ego_pose['translation'], pyquaternion.Quaternion(ego_pose['rotation']), inverse=True)
        v = np.dot(R[:3, :3], np.array([1, 0, 0]))
        yaw = np.arctan2(v[1], v[0])

        # map_size x map_size x 1, 0~255 (uint8)
        img_pedcross = (64.0*self.draw_others(xyz, yaw, x_range, y_range, map_size, scene_location, 'ped_crossing')).astype('uint8')

        # map_size x map_size x 3, 0~255 (float)
        img_centerlines = np.concatenate([img_pedcross, img_pedcross, img_pedcross], axis=2)
        img_centerlines = self.draw_centerlines(img_centerlines, Rinv, xyz, x_range, y_range, map_size, scene_location).astype('float')

        # map_size x map_size x 1, 0~255 (float)
        img_traj = self.draw_agent_trajectories(obs_traj, x_range, y_range, map_size, category)[:, :, 0].reshape(map_size, map_size, 1)

        # cv2.imshow("", img_centerlines.astype('uint8'))
        # cv2.waitKey(0)

        # map_size x map_size x 4, 0~255 (float)
        img = np.concatenate([img_centerlines, img_traj], axis=2)

        return img.astype('float')/255.0

    def make_topview_map_loadertypeF(self, xyz, R, Rinv, scene_location, x_range, y_range, map_size, obs_traj, category):

        # ego-pose
        R = self.transfrom_matrix(R, xyz, inverse=False)
        Rinv = self.transfrom_matrix(Rinv, xyz, inverse=True)
        v = np.dot(R[:3, :3], np.array([1, 0, 0]))
        yaw = np.arctan2(v[1], v[0])

        # map_size x map_size x 1, 0~255 (uint8)
        img_pedcross = (64.0*self.draw_others(xyz, yaw, x_range, y_range, map_size, scene_location, 'ped_crossing')).astype('uint8')

        # map_size x map_size x 3, 0~255 (float)
        img_centerlines = np.concatenate([img_pedcross, img_pedcross, img_pedcross], axis=2)
        img_centerlines = self.draw_centerlines(img_centerlines, Rinv, xyz, x_range, y_range, map_size, scene_location).astype('float')

        # map_size x map_size x 1, 0~255 (float)
        img_traj = self.draw_agent_trajectories(obs_traj, x_range, y_range, map_size, category)[:, :, 0].reshape(map_size, map_size, 1)

        # cv2.imshow("", img_traj.astype('uint8'))
        # cv2.waitKey(0)

        # map_size x map_size x 4, 0~255 (float)
        img_centerlines = np.concatenate([img_centerlines, img_traj], axis=2)

        return img_centerlines.astype('float')/255.0

    def make_topview_map_for_pedestrian(self, ego_pose, scene_location, x_range, y_range, map_size, category):

        # ego-pose
        xyz = ego_pose['translation']
        R = transform_matrix(ego_pose['translation'], pyquaternion.Quaternion(ego_pose['rotation']), inverse=False)
        Rinv = transform_matrix(ego_pose['translation'], pyquaternion.Quaternion(ego_pose['rotation']), inverse=True)
        v = np.dot(R[:3, :3], np.array([1, 0, 0]))
        yaw = np.arctan2(v[1], v[0])

        static_scenes = []

        # drivable space (map_size x map_size x 1)
        img_drivable = self.draw_drivable_space(xyz, yaw, x_range, y_range, map_size, scene_location)
        static_scenes.append(img_drivable)

        # others (map_size x map_size x 1)
        img_pedcross = self.draw_others(xyz, yaw, x_range, y_range, map_size, scene_location, 'ped_crossing')
        img_walkway = self.draw_others(xyz, yaw, x_range, y_range, map_size, scene_location, 'walkway')
        static_scenes.append(img_pedcross)
        static_scenes.append(img_walkway)

        output = np.concatenate(static_scenes, axis=2)
        # cv2.imshow('test', (255*output).astype('uint8'))
        # cv2.waitKey(0)

        return output.astype('float')

    def draw_centerlines(self, img, Rinv, xyz, x_range, y_range, map_size, scene_location):

        # global coord.
        pose_lists = self.centerlines[scene_location]

        w_x_min = xyz[0] + x_range[0] - 10
        w_x_max = xyz[0] + x_range[1] + 10
        w_y_min = xyz[1] + y_range[0] - 10
        w_y_max = xyz[1] + y_range[1] + 10
        win_min_max = (w_x_min, w_y_min, w_x_max, w_y_max)

        for l in range(len(pose_lists)):

            cur_lane = pose_lists[l]
            l_x_max = np.max(cur_lane[:, 0])
            l_x_min = np.min(cur_lane[:, 0])
            l_y_max = np.max(cur_lane[:, 1])
            l_y_min = np.min(cur_lane[:, 1])

            lane_min_max = (l_x_min, l_y_min, l_x_max, l_y_max)

            if (correspondance_check(win_min_max, lane_min_max) == True):
                # global to agent-centric
                cur_lane = self.transform_pc(Rinv, cur_lane.T).T

                # draw
                img = self.draw_lines_on_topview_with_coloryaw(img, cur_lane[:, :2], x_range, y_range, map_size=map_size)
                # img = self.draw_lines_on_topview(img, cur_lane[:, :2], x_range, y_range, map_size, )


        # cv2.imshow("", img.astype('uint8'))
        # cv2.waitKey(0)

        return img.astype('float')

    def draw_centerlines_multich(self, Rinv, xyz, x_range, y_range, map_size, scene_location):

        img_0 = np.zeros(shape=(map_size, map_size, 1))
        img_1 = np.zeros(shape=(map_size, map_size, 1))
        img_2 = np.zeros(shape=(map_size, map_size, 1))
        img_3 = np.zeros(shape=(map_size, map_size, 1))

        # global coord.
        pose_lists = self.centerlines[scene_location]

        w_x_min = xyz[0] + x_range[0] - 10
        w_x_max = xyz[0] + x_range[1] + 10
        w_y_min = xyz[1] + y_range[0] - 10
        w_y_max = xyz[1] + y_range[1] + 10
        win_min_max = (w_x_min, w_y_min, w_x_max, w_y_max)

        for l in range(len(pose_lists)):

            cur_lane = pose_lists[l]
            l_x_max = np.max(cur_lane[:, 0])
            l_x_min = np.min(cur_lane[:, 0])
            l_y_max = np.max(cur_lane[:, 1])
            l_y_min = np.min(cur_lane[:, 1])

            lane_min_max = (l_x_min, l_y_min, l_x_max, l_y_max)

            if (correspondance_check(win_min_max, lane_min_max) == True):
                # global to agent-centric
                cur_lane = self.transform_pc(Rinv, cur_lane.T).T

                # draw
                img_0, img_1, img_2, img_3 = self.draw_lines_on_multich_topview(img_0, img_1, img_2, img_3, cur_lane[:, :2], x_range, y_range, map_size=map_size)

        img = np.concatenate([img_0, img_1, img_2, img_3], axis=2)
        return img.astype('float')


    def draw_agent_trajectories(self, obs_traj, x_range, y_range, map_size, category):

        seq_len, batch, dim = obs_traj.shape
        axis_range_y = y_range[1] - y_range[0]
        axis_range_x = x_range[1] - x_range[0]
        scale_y = float(map_size - 1) / axis_range_y
        scale_x = float(map_size - 1) / axis_range_x

        img = np.zeros(shape=(map_size, map_size, 3))
        for b in range(batch):

            if (category[b] == 0):
                circle_size = 5
            else:
                circle_size = 2


            col_pels = -(obs_traj[:, b, 1] * scale_y).astype(np.int32)
            row_pels = -(obs_traj[:, b, 0] * scale_x).astype(np.int32)

            col_pels += int(np.trunc(y_range[1] * scale_y))
            row_pels += int(np.trunc(x_range[1] * scale_x))

            for j in range(0, seq_len):

                if (np.isnan(col_pels[j])):
                    continue

                brightness = int(255.0 * float(j+1) / float(seq_len+1))
                color = (brightness, brightness, brightness)
                img = cv2.circle(img, (col_pels[j], row_pels[j]), circle_size, color, -1)

        return img


    def draw_target_centerlines(self, img, pose_lists, x_range, y_range, map_size):


        for l in range(len(pose_lists)):

            line_pose = np.concatenate(pose_lists[l], axis=0)

            # draw
            img = self.draw_lines_on_topview_with_coloryaw(img, line_pose[:, :2], x_range, y_range, map_size=map_size)

        # cv2.imshow("", img.astype('uint8'))
        # cv2.waitKey(0)

        return img.astype('float') / 255.0


    def draw_lines_on_topview_with_coloryaw(self, img, line, x_range, y_range, map_size):

        diff = line[1:] - line[:-1]
        line_yaws = calc_yaw_from_points(diff) + np.pi # 0 to 2*pi
        line_angle_deg = line_yaws * 180 / np.pi

        # debug ---
        chk0 = line_angle_deg > 360
        chk1 = line_angle_deg < 0
        assert (np.count_nonzero(chk0) == 0)
        assert (np.count_nonzero(chk1) == 0)
        # debug ---

        axis_range_y = y_range[1] - y_range[0]
        axis_range_x = x_range[1] - x_range[0]
        scale_y = float(map_size - 1) / axis_range_y
        scale_x = float(map_size - 1) / axis_range_x

        col_pels = -(line[:, 1] * scale_y).astype(np.int32)
        row_pels = -(line[:, 0] * scale_x).astype(np.int32)

        col_pels += int(np.trunc(y_range[1] * scale_y))
        row_pels += int(np.trunc(x_range[1] * scale_x))

        for j in range(1, line.shape[0]):

            rgb_n = colorsys.hsv_to_rgb(line_angle_deg[j-1] / 360, 1., 1.)
            color = (int(255*rgb_n[2]), int(255*rgb_n[1]), int(255*rgb_n[0]))

            start = (col_pels[j], row_pels[j])
            end = (col_pels[j-1], row_pels[j-1])
            cv2.line(img, start, end, color, self.centerline_width)

        return img


    def draw_lines_on_multich_topview(self, img_0, img_1, img_2, img_3, line, x_range, y_range, map_size):

        diff = line[1:] - line[:-1]
        line_yaws = calc_yaw_from_points(diff)
        chk = line_yaws < 0
        line_yaws[chk] += 2 * np.pi  # 0 to 2*pi
        line_angle_deg = line_yaws * 180 / np.pi

        axis_range_y = y_range[1] - y_range[0]
        axis_range_x = x_range[1] - x_range[0]
        scale_y = float(map_size - 1) / axis_range_y
        scale_x = float(map_size - 1) / axis_range_x

        col_pels = -(line[:, 1] * scale_y).astype(np.int32)
        row_pels = -(line[:, 0] * scale_x).astype(np.int32)

        col_pels += int(np.trunc(y_range[1] * scale_y))
        row_pels += int(np.trunc(x_range[1] * scale_x))

        for j in range(1, line.shape[0]):

            rgb_n = colorsys.hsv_to_rgb(line_angle_deg[j-1] / 360, 1., 1.)
            color = (int(255*rgb_n[2]), int(255*rgb_n[1]), int(255*rgb_n[0]))

            start = (col_pels[j], row_pels[j])
            end = (col_pels[j-1], row_pels[j-1])

            if (line_angle_deg[j - 1] < 0 + 50.0 or line_angle_deg[j - 1] > 360.0 - 50.0):
                cv2.line(img_0, start, end, (255, 255, 255), 1)

            if (line_angle_deg[j - 1] > 90.0 - 50.0 and line_angle_deg[j - 1] < 90.0 + 50.0):
                cv2.line(img_1, start, end, (255, 255, 255), 1)

            if (line_angle_deg[j - 1] > 180 - 50.0 and line_angle_deg[j - 1] < 180.0 + 50.0):
                cv2.line(img_2, start, end, (255, 255, 255), 1)

            if (line_angle_deg[j - 1] > 270 - 50.0 and line_angle_deg[j - 1] < 270.0 + 50.0):
                cv2.line(img_3, start, end, (255, 255, 255), 1)

        return img_0, img_1, img_2, img_3


    def draw_lines_on_topview(self, img, line, x_range, y_range, map_size, color):

        axis_range_y = y_range[1] - y_range[0]
        axis_range_x = x_range[1] - x_range[0]
        scale_y = float(map_size - 1) / axis_range_y
        scale_x = float(map_size - 1) / axis_range_x

        col_pels = -(line[:, 1] * scale_y).astype(np.int32)
        row_pels = -(line[:, 0] * scale_x).astype(np.int32)

        col_pels += int(np.trunc(y_range[1] * scale_y))
        row_pels += int(np.trunc(x_range[1] * scale_x))

        for j in range(1, line.shape[0]):

            start = (col_pels[j], row_pels[j])
            end = (col_pels[j-1], row_pels[j-1])
            cv2.line(img, start, end, color, self.centerline_width)
            # cv2.line(img, start, end, color, 2)

        return img


    def draw_drivable_space(self, xyz, yaw, x_range, y_range, map_size, scene_location):

        patch_box = (xyz[0], xyz[1], x_range[1]-x_range[0], y_range[1]-y_range[0])
        patch_x = patch_box[0]
        patch_y = patch_box[1]
        patch_angle = np.rad2deg(yaw) + 90
        target_patch = self.nusc_maps[scene_location].explorer.get_patch_coord(patch_box, patch_angle)

        layer_name = 'drivable_area'
        records = getattr(self.nusc_maps[scene_location], layer_name)

        polygon_list = []
        for record in records:
            polygons = [self.nusc_maps[scene_location].extract_polygon(polygon_token) for polygon_token in record['polygon_tokens']]

            for polygon in polygons:
                new_polygon = polygon.intersection(target_patch)
                if not new_polygon.is_empty:
                    new_polygon = affinity.rotate(new_polygon, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
                    new_polygon = affinity.affine_transform(new_polygon, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])

                    if new_polygon.geom_type is 'Polygon':
                        new_polygon = MultiPolygon([new_polygon])
                    polygon_list.append(new_polygon)

        local_box = (0.0, 0.0, patch_box[2], patch_box[3])
        canvas_size = (map_size, map_size)
        map_mask = self.nusc_maps[scene_location].explorer._polygon_geom_to_mask(polygon_list, local_box, layer_name, canvas_size)

        return np.fliplr(map_mask[:, :].reshape(map_size, map_size, 1))


    def draw_road_segment(self, xyz, yaw, x_range, y_range, map_size, scene_location):

        patch_box = (xyz[0], xyz[1], x_range[1]-x_range[0], y_range[1]-y_range[0])
        patch_angle = np.rad2deg(yaw) + 90
        patch_x = patch_box[0]
        patch_y = patch_box[1]
        target_patch = self.nusc_maps[scene_location].explorer.get_patch_coord(patch_box, patch_angle)

        layer_name = 'road_segment'
        records = getattr(self.nusc_maps[scene_location], layer_name)

        polygon_list = []
        polygon_list_intersection = []
        for record in records:
            polygon = self.nusc_maps[scene_location].extract_polygon(record['polygon_token'])

            if polygon.is_valid:
                new_polygon = polygon.intersection(target_patch)
                if not new_polygon.is_empty:
                    new_polygon = affinity.rotate(new_polygon, -patch_angle,
                                                  origin=(patch_x, patch_y), use_radians=False)
                    new_polygon = affinity.affine_transform(new_polygon,
                                                            [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                    if new_polygon.geom_type is 'Polygon':
                        new_polygon = MultiPolygon([new_polygon])

                    if (record['is_intersection']):
                        polygon_list_intersection.append(new_polygon)
                    else:
                        polygon_list.append(new_polygon)

        local_box = (0.0, 0.0, patch_box[2], patch_box[3])
        canvas_size = (map_size, map_size)
        map_mask = self.nusc_maps[scene_location].explorer._polygon_geom_to_mask(polygon_list, local_box, layer_name, canvas_size)
        map_mask_intersection = self.nusc_maps[scene_location].explorer._polygon_geom_to_mask(polygon_list_intersection, local_box, layer_name, canvas_size)

        map_mask = np.fliplr(map_mask[:, :].reshape(map_size, map_size, 1))
        map_mask_intersection = np.fliplr(map_mask_intersection[:, :].reshape(map_size, map_size, 1))

        return np.concatenate([map_mask, map_mask_intersection], axis=2)


    def draw_lane(self, xyz, yaw, x_range, y_range, map_size, scene_location):

        patch_box = (xyz[0], xyz[1], x_range[1]-x_range[0], y_range[1]-y_range[0])
        patch_angle = np.rad2deg(yaw) + 90
        local_box = (0.0, 0.0, patch_box[2], patch_box[3])
        layer_name = 'lane'
        canvas_size = (map_size, map_size)

        patch_x = patch_box[0]
        patch_y = patch_box[1]
        patch = self.nusc_maps[scene_location].explorer.get_patch_coord(patch_box, patch_angle)
        records = getattr(self.nusc_maps[scene_location], layer_name)

        polygon_list = []
        polygon_list_car = []
        for record in records:
            polygon = self.nusc_maps[scene_location].extract_polygon(record['polygon_token'])

            if polygon.is_valid:
                new_polygon = polygon.intersection(patch)
                if not new_polygon.is_empty:
                    new_polygon = affinity.rotate(new_polygon, -patch_angle,
                                                  origin=(patch_x, patch_y), use_radians=False)
                    new_polygon = affinity.affine_transform(new_polygon,
                                                            [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                    if new_polygon.geom_type is 'Polygon':
                        new_polygon = MultiPolygon([new_polygon])

                    if (record['lane_type']=='CAR'):
                        polygon_list_car.append(new_polygon)
                    else:
                        polygon_list.append(new_polygon)

        map_mask = self.nusc_maps[scene_location].explorer._polygon_geom_to_mask(polygon_list, local_box, layer_name, canvas_size)
        map_mask_car = self.nusc_maps[scene_location].explorer._polygon_geom_to_mask(polygon_list_car, local_box, layer_name, canvas_size)

        map_mask = np.fliplr(map_mask[:, :].reshape(map_size, map_size, 1))
        map_mask_car = np.fliplr(map_mask_car[:, :].reshape(map_size, map_size, 1))

        return np.concatenate([map_mask, map_mask_car], axis=2)


    def draw_others(self, xyz, yaw, x_range, y_range, map_size, scene_location, layer_name):

        patch_box = (xyz[0], xyz[1], x_range[1]-x_range[0], y_range[1]-y_range[0])
        patch_angle = np.rad2deg(yaw) + 90
        patch_x = patch_box[0]
        patch_y = patch_box[1]
        target_patch = self.nusc_maps[scene_location].explorer.get_patch_coord(patch_box, patch_angle)

        records = getattr(self.nusc_maps[scene_location], layer_name)

        polygon_list = []
        for record in records:
            polygon = self.nusc_maps[scene_location].extract_polygon(record['polygon_token'])

            if polygon.is_valid:
                new_polygon = polygon.intersection(target_patch)
                if not new_polygon.is_empty:
                    new_polygon = affinity.rotate(new_polygon, -patch_angle,
                                                  origin=(patch_x, patch_y), use_radians=False)
                    new_polygon = affinity.affine_transform(new_polygon,
                                                            [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                    if new_polygon.geom_type is 'Polygon':
                        new_polygon = MultiPolygon([new_polygon])

                    # if (layer_name == 'stop_line'):
                    #     if (record['stop_line_type'] == 'TRAFFIC_LIGHT'):
                    #         polygon_list.append(new_polygon)
                    # else:
                    #     polygon_list.append(new_polygon)

                    if (layer_name == 'stop_line'):
                        if (record['stop_line_type'] == 'TRAFFIC_LIGHT' and len(record['traffic_light_tokens']) > 0):
                            polygon_list.append(new_polygon)
                    else:
                        polygon_list.append(new_polygon)

        local_box = (0.0, 0.0, patch_box[2], patch_box[3])
        canvas_size = (map_size, map_size)
        map_mask = self.nusc_maps[scene_location].explorer._polygon_geom_to_mask(polygon_list, local_box, layer_name, canvas_size)
        return np.fliplr(map_mask[:, :].reshape(map_size, map_size, 1))

    # ------------------------------------------
    # Lane search
    # ------------------------------------------

    def find_best_lane(self, possible_lanes, trajectory, scene_location, obs_len):

        '''
        Input
        possible_lanes : list of 'token sequence list'
        trajectory : global coordinate positions

        Output
        sorted_lanes : list of 'token sequence list' (best matched lane comes first)
        '''

        min_vals = []
        for _, tok_seq in enumerate(possible_lanes):

            path = []
            for __, tok in enumerate(tok_seq):
                lane_record = self.nusc_maps[scene_location].get_arcline_path(tok)
                coords = np.array(discretize_lane(lane_record, resolution_meters=0.25))
                path.append(coords)
            path = np.concatenate(path, axis=0)

            min_val = 0
            for i in range(obs_len, trajectory.shape[0]):
                err = np.sum(np.abs(path[:, :2] - trajectory[i, :2].reshape(1, 2)), axis=1)
                min_val += np.min(err)
            min_vals.append(min_val)

        sorted_lanes = []
        sort_idx = np.argsort(np.array(min_vals))
        for i in range(len(min_vals)):
            sorted_lanes.append(possible_lanes[sort_idx[i]])

        return sorted_lanes

    def find_possible_lanes(self, agent, lidar_now_token, scene_location):


        if (agent.track_id == 'EGO'):
            lidar_now_data = self.nusc.get('sample_data', lidar_now_token)
            pose = self.nusc.get('ego_pose', lidar_now_data['ego_pose_token'])
        else:
            pose = self.nusc.get('sample_annotation', agent.track_id)

        R = transform_matrix(pose['translation'], pyquaternion.Quaternion(pose['rotation']), inverse=False)
        Rinv = transform_matrix(pose['translation'], pyquaternion.Quaternion(pose['rotation']), inverse=True)
        v = np.dot(R[:3, :3], np.array([1, 0, 0]))
        agent_yaw = np.arctan2(v[1], v[0])

        xyz = np.array(pose['translation'])
        x, y, z = xyz

        # find the lanes inside the range
        lanes = self.get_lane_records_in_radius(x, y, scene_location)
        lanes = lanes['lane'] + lanes['lane_connector']

        # remove lane segments with opposite direction [tok, tok, ...]
        target_lanes = self.remove_opposite_directions(copy.deepcopy(lanes), scene_location, xyz, Rinv, agent_yaw)

        # merge connected lanes [[tok,tok, ...], [tok], ...]
        target_path_lists = self.merge_connected_lanes(copy.deepcopy(target_lanes), scene_location)

        # add incoming lane segments
        target_path_lists = self.add_incoming_lanes(copy.deepcopy(target_path_lists), scene_location)

        # find possible outgoing paths
        num_levels = 10
        for l in range(num_levels):
            target_path_lists = self.find_next_level_lanes(copy.deepcopy(target_path_lists), scene_location)

        # prune line segs in paths
        if (len(target_path_lists) > 0):
            target_path_lists = self.prune_lane_segs(copy.deepcopy(target_path_lists), xyz[:2], scene_location)
            target_path_lists = self.remove_overlapped_paths(copy.deepcopy(target_path_lists))


        return target_path_lists

    def prune_lane_segs(self, path_list, xy, scene_location):

        # remove lanes w/o arc path
        path_list_tmp = []
        for i in range(len(path_list)):
            tok_seq = []
            for j in range(len(path_list[i])):
                cur_tok = path_list[i][j]
                if not self.nusc_maps[scene_location].arcline_path_3.get(cur_tok):
                    do_nothing = 0
                else:
                    tok_seq.append(cur_tok)
            if (len(tok_seq) > 0):
                path_list_tmp.append(tok_seq)
        path_list = copy.deepcopy(path_list_tmp)

        out_list = []
        for _, tok_list in enumerate(path_list):

            num_segs = len(tok_list)
            info = np.zeros(shape=(num_segs, 4))
            for __, tok in enumerate(tok_list):

                lane_record = self.nusc_maps[scene_location].get_arcline_path(tok)
                coords = np.array(discretize_lane(lane_record, resolution_meters=0.25))

                # find the closest point
                dist = np.sqrt(np.sum((coords[:, :2] - xy.reshape(1, 2)) ** 2, axis=1))
                min_dist = np.min(dist)
                min_idx = np.argmin(dist)
                max_dist = np.max(dist)
                length = np.sqrt(np.sum((coords[0, :2] - coords[-1, :2]) ** 2))

                info[__, 0] = min_dist
                info[__, 1] = max_dist
                info[__, 2] = length
                info[__, 3] = np.sqrt(np.sum((coords[min_idx, :2] - coords[-1, :2]) ** 2))

            closest_lane_idx = np.argmin(info[:, 0])
            info[closest_lane_idx, 2] = info[closest_lane_idx, 3]

            backward_list = tok_list[:closest_lane_idx]
            forward_list = tok_list[closest_lane_idx:]

            candis = np.argwhere(np.cumsum(info[closest_lane_idx:, 2]) - self.min_forward_len > 0)
            if (candis.shape[0] == 0):
                last_forward_idx = len(forward_list) - 1
            else:
                last_forward_idx = candis[0, 0]

            candis = np.argwhere(np.cumsum(info[:closest_lane_idx, 2][::-1])[::-1] - self.min_backward_len > 0)
            if (candis.shape[0] == 0):
                last_backward_idx = 0
            else:
                last_backward_idx = candis[-1, 0]

            prune_list = backward_list[last_backward_idx:] + forward_list[:last_forward_idx+1]
            out_list.append(prune_list)

        return out_list

    def remove_overlapped_paths(self, path_list):

        out_list = []
        path_list_tmp = copy.deepcopy(path_list)
        while (len(path_list) > 0):
            cp = path_list[0]

            # after this loop, there is no 'cp' in the list
            while(cp in path_list_tmp):
                path_list_tmp.remove(cp)

            out_list.append(cp)
            path_list = copy.deepcopy(path_list_tmp)

        return out_list

    def get_lane_records_in_radius(self, x, y, scene_location, radius=10):
        lanes = self.nusc_maps[scene_location].get_records_in_radius(x, y, radius=radius, layer_names=['lane', 'lane_connector'])
        return lanes

    def remove_opposite_directions(self, lanes, scene_location, xyz, ego_from_global, agent_yaw):


        def _min(a, b):
            if (a >= b):
                return b
            else:
                return a

        def pos_neg_sort(a, b):
            if (a > 0):
                return a, b
            if (b > 0):
                return b, a

        def angle_difference(theta0, theta1):
            '''
            theta in degree
            '''

            if (theta0 == 0):
                theta0 += 0.0001

            if (theta1 == 0):
                theta1 += 0.0001


            # if two have the same sign
            if (theta0*theta1 > 0):
                angle_diff = abs(theta0 - theta1)
                return angle_diff

            else:
                pos_angle, neg_angle = pos_neg_sort(theta0, theta1)
                neg_angle_r = 360 - abs(neg_angle)

                if (pos_angle - neg_angle < 180):
                    angle_diff = pos_angle - neg_angle
                else:
                    angle_diff = abs(pos_angle - neg_angle_r)

                return angle_diff

        target_lanes = []
        if (len(lanes) == 0):
            return target_lanes

        # agent_yaw_deg = np.rad2deg(agent_yaw)

        for _, tok in enumerate(lanes):

            # get lane record
            lane_record = self.nusc_maps[scene_location].get_arcline_path(tok)

            # get coordinates
            coords = np.array(discretize_lane(lane_record, resolution_meters=0.25))


            # find the closest point
            dist = np.sum((coords[:, :2] - np.array(xyz)[:2].reshape(1, 2))**2, axis=1)
            min_idx = np.argmin(dist)
            if (min_idx == 0):
                min_idx = 1

            coords_ego = self.transform_pc(ego_from_global, coords.T).T
            vec = (coords_ego[min_idx, :2] - coords_ego[min_idx - 1, :2]).reshape(1, 2)
            lane_yaw = calc_yaw_from_points(vec)[0]
            lane_yaw_deg = np.rad2deg(lane_yaw)


            # if (angle_difference(agent_yaw_deg, lane_yaw_deg) > (180.0 / 4)):
            #     continue

            if (abs(lane_yaw_deg) > (180.0 / 4)):
                continue

            target_lanes.append(tok)

        return target_lanes

    def merge_connected_lanes(self, path_list, scene_location):

        remaining_lanes = []
        if (len(path_list) == 0):
            return remaining_lanes

        while (len(path_list) > 0):

            cur_tok = path_list[0]
            cur_tok_list = [cur_tok]
            while (1):

                prev_tok_list = self.nusc_maps[scene_location].get_incoming_lane_ids(cur_tok_list[0])
                next_tok_list = self.nusc_maps[scene_location].get_outgoing_lane_ids(cur_tok_list[-1])

                num_added_tok = 0
                if (len(prev_tok_list) > 0):
                    if (prev_tok_list[0] in path_list):
                        cur_tok_list.insert(0, prev_tok_list[0])
                        path_list.remove(prev_tok_list[0])
                        num_added_tok+=1

                if (len(next_tok_list) > 0):
                    if (next_tok_list[0] in path_list):
                        cur_tok_list.insert(len(cur_tok_list), next_tok_list[0])
                        path_list.remove(next_tok_list[0])
                        num_added_tok+=1

                if (num_added_tok == 0):
                    remaining_lanes.append(cur_tok_list)
                    path_list.remove(cur_tok)
                    break


        return remaining_lanes

    def find_next_level_lanes(self, path_list, scene_location):

        path_list_ext = []
        if (len(path_list) == 0):
            return path_list_ext

        for _, cur_tok_list in enumerate(path_list):
            if (cur_tok_list[-1] in self.nusc_maps[scene_location].connectivity):
                next_tok_list = self.nusc_maps[scene_location].get_outgoing_lane_ids(cur_tok_list[-1])
                if (len(next_tok_list) > 0):
                    for _, next_tok in enumerate(next_tok_list):
                        path_list_ext.append(cur_tok_list+ [next_tok])
                else:
                    path_list_ext.append(cur_tok_list)
            else:
                path_list_ext.append(cur_tok_list)


        # final connectivity check
        out_list = []
        for _, cur_tok_list in enumerate(path_list_ext):
            if (cur_tok_list[-1] not in self.nusc_maps[scene_location].connectivity):
                cur_tok_list.pop(len(cur_tok_list)-1)
            out_list.append(cur_tok_list)

        return out_list

    def add_incoming_lanes(self, path_list, scene_location):

        out_list = []
        if (len(path_list) == 0):
            return out_list

        for idx in range(len(path_list)):
            tok_list = path_list[idx]

            tok_m1_list = self.nusc_maps[scene_location].get_incoming_lane_ids(tok_list[0])
            if (len(tok_m1_list) > 0):
                tok_list.insert(0, tok_m1_list[0])

            out_list.append(tok_list)

        return out_list

    def __repr__(self):
        return f"Nuscenes Map Helper."


def in_range_points(points, x, y, z, x_range, y_range, z_range):

    points_select = points[np.logical_and.reduce((x > x_range[0], x < x_range[1], y > y_range[0], y < y_range[1], z > z_range[0], z < z_range[1]))]
    return np.around(points_select, decimals=2)


def correspondance_check(win_min_max, lane_min_max):

    # four points for window and lane box
    w_x_min, w_y_min, w_x_max, w_y_max = win_min_max
    l_x_min, l_y_min, l_x_max, l_y_max = lane_min_max

    w_TL = (w_x_min, w_y_max)  # l1
    w_BR = (w_x_max, w_y_min)  # r1

    l_TL = (l_x_min, l_y_max)  # l2
    l_BR = (l_x_max, l_y_min)  # r2

    # If one rectangle is on left side of other
    # if (l1.x > r2.x | | l2.x > r1.x)
    if (w_TL[0] > l_BR[0] or l_TL[0] > w_BR[0]):
        return False

    # If one rectangle is above other
    # if (l1.y < r2.y || l2.y < r1.y)
    if (w_TL[1] < l_BR[1] or l_TL[1] < w_BR[1]):
        return False

    return True

def calc_yaw_from_points(vec1):

    '''
    vec : seq_len x 2
    '''

    seq_len = vec1.shape[0]

    vec1 = vec1.reshape(seq_len, 2)
    vec2 = np.repeat(np.concatenate([np.ones(shape=(1, 1)), np.zeros(shape=(1, 1))], axis=1), seq_len, 0)

    x1 = vec1[:, 0]
    y1 = vec1[:, 1]
    x2 = vec2[:, 0]
    y2 = vec2[:, 1]

    dot = y1 * y2 + x1 * x2  # dot product
    det = y1 * x2 - x1 * y2  # determinant

    heading = np.arctan2(det, dot)  # -1x because of left side is POSITIVE

    return heading
