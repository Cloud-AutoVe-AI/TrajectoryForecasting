import cv2
import numpy as np

from random import randint
from utils.functions import read_config, print_voxelization_progress
from utils.libraries import *
from ETRIDataset.agent import Agent
from ETRIDataset.scene import Scene
from ETRIDataset.VossHelper import VossHelper, Pose
from ETRIDataset.visualization import *
from ETRIDataset.map import Map
import pathlib

class DatasetPlot:

    def __init__(self, args):

        # nuscenes map api
        self.map = Map(args)

        self.dataset_path = "/home/dooseop/DATASET/voss/"
        self.obs_len = int(args.past_horizon_seconds * args.target_sample_period)
        self.pred_len = int(args.future_horizon_seconds * args.target_sample_period)
        self.sub_step = int(10.0 / args.target_sample_period)
        self.target_sample_period = args.target_sample_period
        self.seq_len = self.obs_len + self.pred_len

        self.min_past_horizon_seconds = args.min_past_horizon_seconds
        self.min_future_horizon_seconds = args.min_future_horizon_seconds
        self.map_size = args.map_size

        self.vs_mode = args.vs_mode
        self.is_show_text = args.is_show_text
        self.speed_thr = args.speed_thr
        self.curvature_thr = args.curvature_thr

        self.folder_path = os.path.join(pathlib.Path(__file__).parent.resolve(), 'captures')
        if (os.path.exists(self.folder_path) == False):
            os.mkdir(self.folder_path)

        # voss helper
        self.vh = VossHelper()

        self.dpi = 80
        self.color_centerline = (0, 0, 0)
        self.color_roadmark = 'yellowgreen'
        self.color_laneside = (0.5, 0.5, 0.5)
        self.color_pallete = [(1, 0, 0), (0, 1, 0), (1, 0.5, 0), (1, 0, 1), (0, 1, 1)]

    def plot_current_scene(self, target_scenes):

        # start frame index
        start_frm_idx = 20

        if (self.vs_mode == 0):

            num_total_trajs = 0

            # half map size
            delta_x = 1500
            delta_y = 1500
            x_range = (-delta_x, delta_x)
            y_range = (-delta_y, delta_y)

            # map center position
            position = np.array([232872.414, 420707.957, 0]).reshape(1, 3)
            ref_pose = Pose(heading=0, position=position)

            # draw whole map
            fig, (ax, axx, axxx) = plt.subplots(1, 3)
            ax = self.draw_centerlines(ax, ref_pose, x_range, y_range, self.map_size)
            ax = self.draw_polygons(ax, ref_pose, x_range, y_range, self.map_size, self.color_roadmark, 0.5)

            speed_pdf = np.zeros(shape=(10))
            speed, curvature = [], []
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
                agent_ids = np.unique(data[:, 2])

                ego_veh = data[data[:, 2] == -1]
                ego_veh_crop = ego_veh[ego_veh[:, 0] > start_frm_idx]

                xy = ref_pose.to_agent(ego_veh_crop[int(ego_veh_crop.shape[0]/2), 3:5])
                if (self.is_show_text == 1):
                    shift_x = 8 * (np.random.rand(1) - 0.5)
                    shift_y = 8 * (np.random.rand(1) - 0.5)
                    self.insert_text_on_topview(ax, xy[0, 0]+shift_x, xy[0, 1]+shift_y, x_range, y_range, self.map_size, scene_name, isRandom=True)

                for _, aid in enumerate(agent_ids):
                    if (aid == -1):
                        continue

                    ngh_veh = data[data[:, 2] == aid]
                    if (ngh_veh[0, 1] > 1):
                        continue

                    ngh_veh_crop = ngh_veh[ngh_veh[:, 0] > start_frm_idx]
                    ngh_veh_crop = ngh_veh_crop[0::self.sub_step] # subsampling

                    for start in range(0, ngh_veh_crop.shape[0], 1):
                        end = start + self.seq_len
                        if (end > ngh_veh_crop.shape[0] - 1):
                            end = ngh_veh_crop.shape[0] - 1

                        if (end - start < self.seq_len):
                            continue

                        target_traj = ngh_veh_crop[start:end, 3:5]
                        num_total_trajs += 1

                        # calc speed
                        disp = (target_traj[1:] - target_traj[:-1])[:, :2]
                        max_speeds_kmph = np.max(3.6 * self.target_sample_period * np.sqrt(np.sum(disp ** 2, axis=1)))

                        speed_idx = int(max_speeds_kmph/10)
                        if (speed_idx > 9):
                            speed_idx = 0
                        speed_pdf[speed_idx] += 1

                        # calc curvature
                        path_len = np.sum(np.sqrt(np.sum(disp ** 2, axis=1)))
                        path_dist = np.sqrt(np.sum((target_traj[0] - target_traj[-1])**2))
                        cur_curvature = path_len / (path_dist + 1e-10)

                        if (max_speeds_kmph < 10):
                            cur_curvature = 1.00001

                        if (max_speeds_kmph > self.speed_thr and cur_curvature > self.curvature_thr):
                            ax = self.draw_lines_on_topview(ax, ref_pose.to_agent(target_traj), x_range, y_range, self.map_size,
                                                                color=(0, 0, 1), alpha=0.5)
                            speed.append(max_speeds_kmph)
                            curvature.append(cur_curvature)


                ax = self.draw_lines_on_topview(ax, ref_pose.to_agent(ego_veh_crop[:, 3:5]), x_range, y_range, self.map_size,
                                                color=(1, 0, 0), alpha=1.0)

            # plt.show()
            # img = self.fig_to_nparray(fig, ax, self.map_size)
            # file_name = 'mode%d_img_%s.png' % (self.vs_mode, data_type)
            # cv2.imwrite(os.path.join(self.folder_path, file_name), img)

            # print(speed_pdf / np.sum(speed_pdf))
            # print(np.cumsum(speed_pdf / np.sum(speed_pdf)))
            pdf = speed_pdf / np.sum(speed_pdf)
            cdf = np.cumsum(pdf)

            axx.scatter(speed, curvature, color=(0, 0, 1), alpha=0.1)
            axx.axhline(y=1.01, color='r', linestyle='-')
            axx.axvline(x=10.0, color='r', linestyle='-')
            axx.axis([00, 100, 0.9, 3])
            axx.set_xlabel("speed (kmph)")
            axx.set_ylabel("curvature (>1)")


            label = ['<10', '<20', '<30', '<40', '<50', '<60', '<70', '<80', '<90', '<100']
            index = np.arange(len(label))
            axxx.bar(index, speed_pdf / np.sum(speed_pdf))
            axxx.set_xticks(index)
            axxx.set_xticklabels(label, fontsize=7)
            axxx.set_xlabel("speed (kmph)")
            axxx.set_ylabel("normalized count")

            ax.set_title('Trajectory Visualization (speed thr : %.1f, curvature thr : %.3f, num trajs : %d)' % (self.speed_thr, self.curvature_thr, num_total_trajs))
            axx.set_title('speed - curvature plot')
            axxx.set_title('speed  distribution (num trajs : %d)' % num_total_trajs)
            plt.show()


        elif (self.vs_mode == 1):

            for scene_name in tqdm(target_scenes):

                num_total_trajs = 0

                fig, (ax, axx, axxx) = plt.subplots(1, 3)
                speed, curvature = [], []
                speed_pdf = np.zeros(shape=(10))

                target_scene_path = os.path.join(self.dataset_path, scene_name)

                '''
                ------------------------------------------------------------------------------------------------
                frame index| class | obj_id | x[m] | y[m] |  z[m]  | heading[rad] | width[m] | length[m] | height[m] | lidar | image
                ------------------------------------------------------------------------------------------------
                0          | 1     | 2      | 3    | 4    |   5    |      6       |    7     | 8         | 9         | 10     | 11
                ------------------------------------------------------------------------------------------------
                '''
                data = self.vh.read_raw_file(os.path.join(target_scene_path, 'label_ori.csv'))[1:]
                agent_ids = np.unique(data[:, 2])

                ego_veh = data[data[:, 2] == -1]
                ego_veh_crop = ego_veh[ego_veh[:, 0] > start_frm_idx]

                center_x = ego_veh_crop[int(ego_veh_crop.shape[0] / 2), 3]
                center_y = ego_veh_crop[int(ego_veh_crop.shape[0] / 2), 4]

                # map center position
                position = np.array([center_x, center_y, 0]).reshape(1, 3)
                ref_pose = Pose(heading=0, position=position)

                # half map size
                delta_x = 60
                delta_y = 60
                x_range = (-delta_x, delta_x)
                y_range = (-delta_y, delta_y)

                # draw whole map
                ax = self.draw_centerlines(ax, ref_pose, x_range, y_range, self.map_size)
                ax = self.draw_polygons(ax, ref_pose, x_range, y_range, self.map_size, self.color_roadmark, 0.5)

                xy = ref_pose.to_agent(ego_veh_crop[int(ego_veh_crop.shape[0]/2), 3:5])
                if (self.is_show_text == 1):
                    shift_x = 8 * (np.random.rand(1) - 0.5)
                    shift_y = 8 * (np.random.rand(1) - 0.5)
                    self.insert_text_on_topview(ax, xy[0, 0]+shift_x, xy[0, 1]+shift_y, x_range, y_range, self.map_size, scene_name, isRandom=True)

                for _, aid in enumerate(agent_ids):
                    if (aid == -1):
                        continue

                    ngh_veh = data[data[:, 2] == aid]
                    if (ngh_veh[0, 1] > 1):
                        continue

                    ngh_veh_crop = ngh_veh[ngh_veh[:, 0] > start_frm_idx]
                    ngh_veh_crop = ngh_veh_crop[0::self.sub_step] # subsampling

                    for start in range(0, ngh_veh_crop.shape[0], 1):
                        end = start + self.seq_len
                        if (end > ngh_veh_crop.shape[0] - 1):
                            end = ngh_veh_crop.shape[0] - 1

                        if (end - start < self.seq_len):
                            continue

                        target_traj = ngh_veh_crop[start:end, 3:5]
                        num_total_trajs+=1

                        # calc speed
                        disp = (target_traj[1:] - target_traj[:-1])[:, :2]
                        max_speeds_kmph = np.max(3.6 * self.target_sample_period * np.sqrt(np.sum(disp ** 2, axis=1)))

                        speed_idx = int(max_speeds_kmph/10)
                        if (speed_idx > 9):
                            speed_idx = 0
                        speed_pdf[speed_idx] += 1

                        # calc curvature
                        path_len = np.sum(np.sqrt(np.sum(disp ** 2, axis=1)))
                        path_dist = np.sqrt(np.sum((target_traj[0] - target_traj[-1])**2))
                        cur_curvature = path_len / (path_dist + 1e-10)

                        if (max_speeds_kmph < 10):
                            cur_curvature = 1.00001

                        if (max_speeds_kmph > self.speed_thr and cur_curvature > self.curvature_thr):
                            ax = self.draw_lines_on_topview(ax, ref_pose.to_agent(target_traj), x_range, y_range, self.map_size,
                                                                color=(0, 0, 1), alpha=0.5)

                            speed.append(max_speeds_kmph)
                            curvature.append(cur_curvature)

                ax = self.draw_lines_on_topview(ax, ref_pose.to_agent(ego_veh_crop[:, 3:5]), x_range, y_range, self.map_size,
                                                color=(1, 0, 0), alpha=1.0, wCircle=False)

                axx.scatter(speed, curvature, color=(0, 0, 1), alpha=0.1)
                axx.axhline(y=1.01, color='r', linestyle='-')
                axx.axvline(x=10.0, color='r', linestyle='-')
                axx.axis([00, 100, 0.9, 3])
                axx.set_xlabel("speed (kmph)")
                axx.set_ylabel("curvature (>1)")

                label = ['<10', '<20', '<30', '<40', '<50', '<60', '<70', '<80', '<90', '<100']
                index = np.arange(len(label))
                axxx.bar(index, speed_pdf / np.sum(speed_pdf))
                axxx.set_xticks(index)
                axxx.set_xticklabels(label, fontsize=7)
                axxx.set_xlabel("speed (kmph)")
                axxx.set_ylabel("normalized count")

                ax.set_title('Trajectory Visualization (speed thr : %.1f, curvature thr : %.3f, num trajs : %d)' % (self.speed_thr, self.curvature_thr, num_total_trajs))
                axx.set_title('speed - curvature plot')
                axxx.set_title('speed  distribution (num trajs : %d)' % num_total_trajs)
                plt.show()

            # img = self.fig_to_nparray(fig, ax, self.map_size)
            # file_name = 'mode%d_img_%s.png' % (self.vs_mode, data_type)
            # cv2.imwrite(os.path.join(self.folder_path, file_name), img)
        else:
            print("[Error] Mode type %d is not supported .." % (self.vs_mode))
            sys.exit(0)


    def insert_text_on_topview(self, ax, x, y, x_range, y_range, map_size, text, fontsize=16, color=(1,0,0), isRandom=False):

        axis_range_y = y_range[1] - y_range[0]
        axis_range_x = x_range[1] - x_range[0]
        scale_y = float(map_size - 1) / axis_range_y
        scale_x = float(map_size - 1) / axis_range_x

        col_pels = -(y * scale_y).astype(np.int32)
        row_pels = -(x * scale_x).astype(np.int32)

        col_pels += int(np.trunc(y_range[1] * scale_y))
        row_pels += int(np.trunc(x_range[1] * scale_x))

        if (isRandom):
            color = self.color_pallete[randint(0, 4)]

        ax.text(col_pels, map_size - row_pels, text, fontsize=fontsize, color=color)
        return ax

    def draw_lines_on_topview(self, ax, line, x_range, y_range, map_size, color, alpha=1.0, wCircle=False):

        axis_range_y = y_range[1] - y_range[0]
        axis_range_x = x_range[1] - x_range[0]
        scale_y = float(map_size - 1) / axis_range_y
        scale_x = float(map_size - 1) / axis_range_x

        col_pels = -(line[:, 1] * scale_y).astype(np.int32)
        row_pels = -(line[:, 0] * scale_x).astype(np.int32)

        col_pels += int(np.trunc(y_range[1] * scale_y))
        row_pels += int(np.trunc(x_range[1] * scale_x))

        if (wCircle):
            ax.plot(col_pels, map_size - row_pels, 'o-', linewidth=1.0, color=color, alpha=alpha)
        else:
            ax.plot(col_pels, map_size - row_pels, '-', linewidth=1.0, color=color, alpha=alpha)
        return ax

    def draw_centerlines(self, ax, pose, x_range, y_range, map_size):

        lane_segs, IDs = self.map.return_centerlines(pose, x_range, y_range)
        for l in range(len(lane_segs)):
            cur_lane = lane_segs[l]
            ax = self.draw_lines_on_topview(ax, cur_lane[:, :2], x_range, y_range, map_size=map_size, color=self.color_centerline)

        return ax

    def draw_polygons(self, ax, pose, x_range, y_range, map_size, facecolor, alpha):

        axis_range_y = y_range[1] - y_range[0]
        axis_range_x = x_range[1] - x_range[0]
        scale_y = float(map_size - 1) / axis_range_y
        scale_x = float(map_size - 1) / axis_range_x

        poly_segs = self.map.return_roadmark(pose, x_range, y_range)

        for i in range(len(poly_segs)):
            cur_polygons = poly_segs[i]

            col_pels = -(cur_polygons[:, 1] * scale_y).astype(np.int32)
            row_pels = -(cur_polygons[:, 0] * scale_x).astype(np.int32)

            col_pels += int(np.trunc(y_range[1] * scale_y))
            row_pels += int(np.trunc(x_range[1] * scale_x))

            col_pels[col_pels < 0] = 0
            col_pels[col_pels > map_size - 1] = map_size - 1

            row_pels[row_pels < 0] = 0
            row_pels[row_pels > map_size - 1] = map_size - 1

            contours = np.concatenate(
                [col_pels.reshape(cur_polygons.shape[0], 1), map_size - row_pels.reshape(cur_polygons.shape[0], 1)], axis=1)

            ax.add_patch(
                patches.Polygon(
                    contours,
                    closed=True,
                    facecolor=facecolor,
                    alpha=alpha
                ))

        #lightgray
        return ax

    def fig_to_nparray(self, fig, ax, map_size):

        fig.set_size_inches(map_size / self.dpi, map_size / self.dpi)
        ax.set_axis_off()

        fig.canvas.draw()
        render_fig = np.array(fig.canvas.renderer._renderer)

        final_img = np.zeros_like(render_fig[:, :, :3]) # 450, 1600
        final_img[:, :, 2] = render_fig[:, :, 0]
        final_img[:, :, 1] = render_fig[:, :, 1]
        final_img[:, :, 0] = render_fig[:, :, 2]

        plt.close()

        return final_img

def main():

    abspath = os.path.dirname(os.path.realpath(__file__))
    os.chdir(Path(abspath).parent.absolute())

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', type=str, default='')
    parser.add_argument('--dataset_type', type=str, default='ETRI')
    parser.add_argument('--past_horizon_seconds', type=float, default=2)
    parser.add_argument('--future_horizon_seconds', type=float, default=4)
    parser.add_argument('--min_past_horizon_seconds', type=float, default=1.5)
    parser.add_argument('--min_future_horizon_seconds', type=float, default=3)
    parser.add_argument('--target_sample_period', type=float, default=2)  # Hz ---

    parser.add_argument('--limit_range', type=int, default=50)
    parser.add_argument('--x_range_min', type=float, default=-90)
    parser.add_argument('--x_range_max', type=float, default=90)
    parser.add_argument('--y_range_min', type=float, default=-90)
    parser.add_argument('--y_range_max', type=float, default=90)
    parser.add_argument('--z_range_min', type=float, default=-1)
    parser.add_argument('--z_range_max', type=float, default=4)
    parser.add_argument('--num_lidar_sweeps', type=int, default=5)
    parser.add_argument('--map_size', type=int, default=4096)

    parser.add_argument('--is_shift_lane_deg', type=int, default=0)
    parser.add_argument('--is_show_text', type=int, default=1)

    # vs_mode
    # 0 (overall traj on entire map)
    # 1 (target dataset traj on corresponding map)
    parser.add_argument('--vs_mode', type=int, default=0)
    parser.add_argument('--speed_thr', type=float, default=0)
    parser.add_argument('--curvature_thr', type=float, default=1.00)

    # 1 ~ 1.01
    # 1.01 ~ 1.05
    # 1.05 ~

    args = parser.parse_args()
    DP = DatasetPlot(args)

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
    train_scenes = R1 + R2 + R3 + R4 + R6 + R7 + R8 + R9 + R10 + R11 + R12 + R15 + R16
    DP.plot_current_scene(target_scenes=train_scenes)



    seungjeok = ['0106', '0107', '0140', '0141', '0142'] # 141
    science_cross = ['0109', '0110', '0111', '0123', '0125', '0126'] # 125
    gusung = ['0135', '0136', '0137'] # 135
    etri_front = ['0143', '0144', '0145'] # 145
    # DP.plot_current_scene(target_scenes=etri_front)

    # test_scenes = ['0011', '0016', '0020', '0036', '0067', '0141', '0125', '0135', '0145']
    # DP.plot_current_scene(target_scenes=test_scenes)


if __name__ == '__main__':
    main()

