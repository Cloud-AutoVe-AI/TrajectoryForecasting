import numpy as np
import math
import random
import sys
import pickle
import csv
import os
import time
import matplotlib.pyplot as plt
import cv2
import copy
from random import randint
import argparse

class ProcessedDataLoader:

    def __init__(self, save_path='', map_size=1080, range_meter=80, obs_len=10, pred_len=20, num_fake_agents=None):

        '''
        ave_path:
        map_size:
        range_meter:
        rotate_angle:
        obs_len:
        pred_len:
        '''

        self.save_path = save_path
        self.map_size = map_size
        self.range_meter = range_meter
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.min_traj_len = obs_len
        self.num_fake_agents = num_fake_agents

        # load preprocessed data
        if (os.path.exists(os.path.join(self.save_path, 'label.csv'))):
            self.data = self.read_raw_file(os.path.join(self.save_path, 'label.csv'))[1:]
        else:
            message = '>> {%s} does not exist!' % os.path.join(self.save_path, 'label.csv')
            sys.exit(message)

        self.read_calib_info()

        self.frm_idx = np.unique(self.data[:, 0])

        # define color palette
        self._create_palette(num_colors=1000)

    def read_calib_info(self):

        # read calib info
        file_name = 'cam0_extcalib.csv'
        file_path = os.path.join(self.save_path, file_name)

        try:
            f = open(file_path)
            reader = csv.reader(f, delimiter=',')
            cnt = 0
            for row in reader:
                if (cnt == 0):
                    K = np.array(row).astype('float').reshape(3, 4)
                else:
                    Rt = np.array(row).astype('float').reshape(4, 4)
                cnt += 1
            f.close()
            self.K, self.Rt = K, Rt

        except:
            self.K, self.Rt = None, None

    def read_raw_file(self, file_dir):

        '''
        ------------------------------------------------------------------------------------------------
        frame index| class | obj_id | x[m] | y[m] |  z[m]  | heading[rad] | width[m] | length[m] | height[m] | lidar | image
        ------------------------------------------------------------------------------------------------
        0          | 1     | 2      | 3    | 4    |   5    |      6       |    7     | 8         | 9         | 10     | 11
        ------------------------------------------------------------------------------------------------
        '''

        return np.genfromtxt(file_dir, delimiter=',')

    def read_point_cloud(self, file_name):
        lidar_filepath = os.path.join(self.save_path, "OUSTER0/%s" % file_name)
        scan = np.fromfile(lidar_filepath, dtype=np.float32)
        return scan.reshape([-1, 4])

    def read_front_image(self, file_name, isresize=True):
        image_filepath = os.path.join(self.save_path, "00/%s" % file_name)
        img = cv2.imread(image_filepath)
        if (isresize):
            img = cv2.resize(img, (320, 240), interpolation=cv2.INTER_CUBIC)

        return img

    def draw_current_scene_topview(self, frm_idx, seq_data, map_size, range_meter, rotate_angle, show_traj=True):

        '''

        seq_data : seq_len x num_agents x feat_dim

        "feat_dim" info
        ------------------------------------------------------------------------------------------------
        frame index| class | obj_id | x[m] | y[m] |  z[m]  | heading[rad] | width[m] | length[m] | height[m] | lidar | image
        ------------------------------------------------------------------------------------------------
        0          | 1     | 2      | 3    | 4    |   5    |      6       |    7     | 8         | 9         | 10     | 11
        ------------------------------------------------------------------------------------------------
        '''

        def draw_circle(img, obj_idx, x, y, color, x_range, y_range, map_size, isempty=True, iswrite=False):

            # for displaying images
            axis_range_y = y_range[1] - y_range[0]
            axis_range_x = x_range[1] - x_range[0]
            scale_y = float(map_size - 1) / axis_range_y
            scale_x = float(map_size - 1) / axis_range_x

            # GT trajs --------------------------------------
            col_pel = -(y * scale_y).astype(np.int32)
            row_pel = -(x * scale_x).astype(np.int32)

            col_pel += int(np.trunc(y_range[1] * scale_y))
            row_pel += int(np.trunc(x_range[1] * scale_x))

            if (isempty):
                img = cv2.circle(img, (col_pel, row_pel), 3, color, 1)
            else:
                img = cv2.circle(img, (col_pel, row_pel), 3, color, -1)

            if (iswrite):
                text = 'ID : ' + str(obj_idx)
                img = cv2.putText(img, text, (col_pel, row_pel), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            return img

        def draw_point_cloud(point_cloud, x_range, y_range, z_range, map_size):

            # rotate points according to car heading
            # points_rot = rotate_points_cloud(points[:, 0:3])
            points_rot = np.copy(point_cloud[:, 0:3])

            # draw on image
            x = points_rot[:, 0]  # same direction to car' heading -> row axis for image display
            y = points_rot[:, 1]  # perpendicular to x direction   -> col axis for image display
            z = points_rot[:, 2]  # perpendicular to ground-plane

            # extract in-range points
            x_lim = in_range_points(x, x, y, z, x_range, y_range, z_range)
            y_lim = in_range_points(y, x, y, z, x_range, y_range, z_range)

            # for displaying images
            row_axis_range = int((x_range[1] - x_range[0]))
            scale = (map_size - 1) / row_axis_range
            col_img = -(y_lim * scale).astype(np.int32)
            row_img = -(x_lim * scale).astype(np.int32)

            # shift negative points to positive points (shift minimum value to 0)
            col_img += int(np.trunc(y_range[1] * scale))
            row_img += int(np.trunc(x_range[1] * scale))

            img = np.zeros((map_size, map_size, 3), np.uint8)
            ch0 = np.copy(img[:, :, 0])
            ch1 = np.copy(img[:, :, 1])
            ch2 = np.copy(img[:, :, 2])

            ch0[row_img.astype('int32'), col_img.astype('int32')] = 256 * np.ones_like(x_lim)
            ch1[row_img.astype('int32'), col_img.astype('int32')] = 128 * np.ones_like(x_lim)
            ch2[row_img.astype('int32'), col_img.astype('int32')] = 128 * np.ones_like(x_lim)

            img[:, :, 0] = ch0.astype('uint8')
            img[:, :, 1] = ch1.astype('uint8')
            img[:, :, 2] = ch2.astype('uint8')

            return img

        def draw_bbox(img, bbox, color, x_range, y_range, map_size):

            '''

               (bottom)          (up)

                front              front
            b0 -------- b3    b4 -------- b7
               |      |          |      |
               |      |          |      |
               |      |          |      |
               |      |          |      |
            b1 -------- b2    b5 -------- b6
                 rear              rear
            '''

            # for displaying images
            axis_range_y = y_range[1] - y_range[0]
            axis_range_x = x_range[1] - x_range[0]
            scale_y = float(map_size - 1) / axis_range_y
            scale_x = float(map_size - 1) / axis_range_x

            # GT trajs --------------------------------------
            col_pels = -(bbox[:, 1] * scale_y).astype(np.int32)
            row_pels = -(bbox[:, 0] * scale_x).astype(np.int32)

            col_pels += int(np.trunc(y_range[1] * scale_y))
            row_pels += int(np.trunc(x_range[1] * scale_x))

            img = cv2.line(img, (col_pels[0], row_pels[0]), (col_pels[1], row_pels[1]), color, 1)
            img = cv2.line(img, (col_pels[1], row_pels[1]), (col_pels[2], row_pels[2]), color, 1)
            img = cv2.line(img, (col_pels[3], row_pels[3]), (col_pels[2], row_pels[2]), color, 1)
            img = cv2.line(img, (col_pels[3], row_pels[3]), (col_pels[0], row_pels[0]), color, 1)
            img = cv2.line(img, (col_pels[1], row_pels[1]), (int((col_pels[3]+col_pels[0])/2), int((row_pels[3]+row_pels[0])/2)), color, 1)
            img = cv2.line(img, (col_pels[2], row_pels[2]), (int((col_pels[3] + col_pels[0]) / 2), int((row_pels[3] + row_pels[0]) / 2)), color, 1)

            return img

        if (self.map_size != map_size):
            self.map_size = map_size

        if (self.range_meter != range_meter):
            self.range_meter = range_meter

        x_range = (-1 * self.range_meter/2, self.range_meter/2)
        y_range = (-1 * self.range_meter/2, self.range_meter/2)

        # draw top-view lidar image
        lidar_file_name = '%08d.bin' % int(seq_data[self.obs_len-1, 0, 10])
        point_cloud = self.read_point_cloud(lidar_file_name)
        img = draw_point_cloud(point_cloud, x_range, y_range, (-3, 1), self.map_size)

        # ego pose
        cur_scene = seq_data[self.obs_len-1]
        heading = cur_scene[cur_scene[:, 2] == -1, :][0, 6]
        R_e2g = make_rot_matrix(heading, 1) # ego-centric to global coordinate
        R_g2e = np.linalg.inv(R_e2g) # global to ego-centric coordinate
        trans_g = cur_scene[cur_scene[:, 2] == -1, :][0, 3:5].reshape(1, 2) # global position


        # for all obj
        num_obj = 0
        for i in range(seq_data.shape[1]):

            # global coord
            cur_obj_traj = seq_data[:, i, :]
            obj_idx = cur_obj_traj[self.obs_len-1, 2]

            # transform to ego-centric
            trans = cur_obj_traj[:, 3:5] - trans_g # seq_len x 2
            cur_obj_traj_e = np.matmul(R_g2e[0], trans.T).T

            # draw bbox
            if (obj_idx != -1):

                '''
                "width, length, height -> 3d bbox"

                (bottom)          (up)
                    front             front            
                b0 -------- b3    b4 -------- b7
                   |      |          |      |
                   |      |          |      |
                   |      |          |      |
                   |      |          |      |
                b1 -------- b2    b5 -------- b6
                     rear              rear
                '''

                # bbox
                w, l, h = cur_obj_traj[self.obs_len-1, 7:10]
                corner_b0 = np.array([l / 2, w / 2]).reshape(1, 2)
                corner_b1 = np.array([-l / 2, w / 2]).reshape(1, 2)
                corner_b2 = np.array([-l / 2, -w / 2]).reshape(1, 2)
                corner_b3 = np.array([l / 2, -w / 2]).reshape(1, 2)
                box_bot = np.concatenate([corner_b0, corner_b1, corner_b2, corner_b3], axis=0)  # 4 x 2

                # agent to global coord
                heading_a = cur_obj_traj[self.obs_len-1, 6]
                R_a2g = make_rot_matrix(heading_a, 1)[0]
                trans_g_a = cur_obj_traj[self.obs_len-1, 3:5].reshape(1, 2)
                box_bot_g = np.matmul(R_a2g, box_bot.T).T + trans_g_a

                # global to ego centric coord
                trans = box_bot_g - trans_g # seq_len x 2
                box_bot_e = np.matmul(R_g2e[0], trans.T).T

                # draw current position
                if (obj_idx == -1):
                    color = (0, 0, 255)
                else:
                    bb_, bg_, br_ = self.palette[int(obj_idx % 999)]
                    color = (bb_, bg_, br_)

                img = draw_bbox(img, box_bot_e, color, x_range, y_range, map_size)

            # draw positions
            for t in range(seq_data.shape[0]):

                x_pos = cur_obj_traj_e[t, 0]
                y_pos = cur_obj_traj_e[t, 1]

                if (np.isnan(x_pos)):
                    continue

                # draw current position
                if (obj_idx == -1):
                    color = (0, 0, 255)
                else:
                    bb_, bg_, br_ = self.palette[int(obj_idx % 999)]
                    color = (bb_, bg_, br_)

                img = draw_circle(img, obj_idx, x_pos, y_pos, color, x_range, y_range, self.map_size, isempty=True, iswrite=False)
                if (t == self.obs_len-1):
                    img = draw_circle(img, obj_idx, x_pos, y_pos, color, x_range, y_range, self.map_size, isempty=False, iswrite=True)

            num_obj+=1

        # --------------------------
        # Write information
        # --------------------------
        text = 'Frm Idx: %05d' % frm_idx
        img = cv2.putText(img, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        text = 'Num Obj: %02d' % num_obj
        img = cv2.putText(img, text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # --------------------------
        # Show front view
        # --------------------------
        image_file_name = '%08d.png' % int(seq_data[self.obs_len-1, 0, 11])
        front_img = self.read_front_image(image_file_name)
        img[-240:, -320:, :] = front_img

        return img

    def project_bbox3d_to_front_img(self, seq_data):
        '''

        seq_data : seq_len x num_agents x feat_dim

        "feat_dim" info
        ------------------------------------------------------------------------------------------------
        frame index| class | obj_id | x[m] | y[m] |  z[m]  | heading[rad] | width[m] | length[m] | height[m] | lidar | image
        ------------------------------------------------------------------------------------------------
        0          | 1     | 2      | 3    | 4    |   5    |      6       |    7     | 8         | 9         | 10     | 11
        ------------------------------------------------------------------------------------------------
        '''


        def draw_bbox_line(img, corner1, corner2, color, thickness=1):

            # if (corner1[1] > 480 - 1):
            #     return img
            #
            # if (corner2[1] > 480 - 1):
            #     return img

            img = cv2.line(img, (corner1[0], corner1[1]), (corner2[0], corner2[1]), color, thickness)

            return img

        def draw_bbox_on_cam(img, bbox3d, K, Rt, color):

            num_corner = bbox3d.shape[0]
            bbox3d = np.concatenate([bbox3d, np.ones(shape=(num_corner, 1))], axis=1)

            bbox_img = []
            for m in range(bbox3d.shape[0]):
                A = np.matmul(np.linalg.inv(Rt), bbox3d[m, :].reshape(4, 1))
                B = np.matmul(K, A)

                x_cur = int(B[0, 0] / B[2, 0])
                y_cur = int(B[1, 0] / B[2, 0])

                bbox_img.append([x_cur, y_cur])

            '''
            "width, length, height -> 3d bbox"

                (bottom)          (up)
                    front             front            
                b0 -------- b3    b4 -------- b7
                   |      |          |      |
                   |      |          |      |
                   |      |          |      |
                   |      |          |      |
                b1 -------- b2    b5 -------- b6
                     rear              rear
            '''

            img = draw_bbox_line(img, bbox_img[0], bbox_img[1], color)
            img = draw_bbox_line(img, bbox_img[1], bbox_img[2], color)
            img = draw_bbox_line(img, bbox_img[2], bbox_img[3], color)
            img = draw_bbox_line(img, bbox_img[3], bbox_img[0], color, thickness=3)

            img = draw_bbox_line(img, bbox_img[4], bbox_img[5], color)
            img = draw_bbox_line(img, bbox_img[5], bbox_img[6], color)
            img = draw_bbox_line(img, bbox_img[6], bbox_img[7], color)
            img = draw_bbox_line(img, bbox_img[7], bbox_img[4], color, thickness=3)

            img = draw_bbox_line(img, bbox_img[0], bbox_img[4], color, thickness=3)
            img = draw_bbox_line(img, bbox_img[1], bbox_img[5], color)
            img = draw_bbox_line(img, bbox_img[2], bbox_img[6], color)
            img = draw_bbox_line(img, bbox_img[3], bbox_img[7], color, thickness=3)

            return img

        image_file_name = '%08d.png' % int(seq_data[self.obs_len-1, 0, 11])
        img = self.read_front_image(image_file_name, isresize=False)

        # ego pose
        cur_scene = seq_data[self.obs_len-1]
        heading = cur_scene[cur_scene[:, 2] == -1, :][0, 6]
        R_e2g = np.eye(3)
        R_e2g[:2, :2] = make_rot_matrix(heading, 1)[0] # ego-centric to global coordinate
        R_g2e = np.linalg.inv(R_e2g) # global to ego-centric coordinate
        trans_g = cur_scene[cur_scene[:, 2] == -1, :][0, 3:6].reshape(1, 3) # global position


        # for all obj
        for i in range(seq_data.shape[1]):

            # global coord
            cur_obj_traj = seq_data[:, i, :]
            obj_idx = cur_obj_traj[self.obs_len-1, 2]

            # transform to ego-centric
            trans = cur_obj_traj[:, 3:6] - trans_g # seq_len x 3
            cur_obj_traj_e = np.matmul(R_g2e, trans.T).T

            x, y = cur_obj_traj_e[self.obs_len-1, :2]
            thr = np.deg2rad(45.0)
            if (np.arctan2(y, x) > thr or np.arctan2(y, x) < -1*thr):
                continue

            # draw bbox
            if (obj_idx != -1):

                '''
                "width, length, height -> 3d bbox"

                (bottom)          (up)
                    front             front            
                b0 -------- b3    b4 -------- b7
                   |      |          |      |
                   |      |          |      |
                   |      |          |      |
                   |      |          |      |
                b1 -------- b2    b5 -------- b6
                     rear              rear
                '''

                # bbox
                w, l, h = cur_obj_traj[self.obs_len-1, 7:10]
                corner_b0 = np.array([l / 2, w / 2, 0]).reshape(1, 3)
                corner_b1 = np.array([-l / 2, w / 2, 0]).reshape(1, 3)
                corner_b2 = np.array([-l / 2, -w / 2, 0]).reshape(1, 3)
                corner_b3 = np.array([l / 2, -w / 2, 0]).reshape(1, 3)
                box_bot = np.concatenate([corner_b0, corner_b1, corner_b2, corner_b3], axis=0)  # 4 x 3

                corner_b4 = np.array([l / 2, w / 2, h]).reshape(1, 3)
                corner_b5 = np.array([-l / 2, w / 2, h]).reshape(1, 3)
                corner_b6 = np.array([-l / 2, -w / 2, h]).reshape(1, 3)
                corner_b7 = np.array([l / 2, -w / 2, h]).reshape(1, 3)
                box_up = np.concatenate([corner_b4, corner_b5, corner_b6, corner_b7], axis=0)  # 4 x 3

                box = np.concatenate([box_bot, box_up], axis=0)

                # agent to global coord1
                heading_a = cur_obj_traj[self.obs_len-1, 6]
                R_a2g = np.eye(3)
                R_a2g[:2, :2] = make_rot_matrix(heading_a, 1)[0]
                trans_g_a = cur_obj_traj[self.obs_len-1, 3:6].reshape(1, 3)
                box_g = np.matmul(R_a2g, box.T).T + trans_g_a

                # global to ego centric coord
                trans = box_g - trans_g # seq_len x 3
                box_e = np.matmul(R_g2e, trans.T).T

                # draw current position
                if (obj_idx == -1):
                    color = (0, 0, 255)
                else:
                    bb_, bg_, br_ = self.palette[int(obj_idx % 999)]
                    color = (bb_, bg_, br_)

                img = draw_bbox_on_cam(img, box_e, self.K, self.Rt, color)

        return img

    def _create_palette(self, num_colors):
        self.palette = []
        for i in range(num_colors):

            br = randint(64, 255)
            bg = randint(64, 255)
            bb = randint(64, 255)

            self.palette.append([bb, bg, br])

    def get_current_scene(self, curr_frm_idx):

        '''
        "feat_dim" info
        ------------------------------------------------------------------------------------------------
        frame index| class | obj_id | x[m] | y[m] |  z[m]  | heading[rad] | width[m] | length[m] | height[m] | lidar | image
        ------------------------------------------------------------------------------------------------
        0          | 1     | 2      | 3    | 4    |   5    |      6       |    7     | 8         | 9         | 10     | 11
        ------------------------------------------------------------------------------------------------
        '''

        feat_dim = self.data.shape[1]

        curr_data = self.data[self.data[:, 0] == curr_frm_idx]
        vids = np.unique(curr_data[:, 2]).tolist()

        trajs = []
        for i in range(0, len(vids)):
            target_vid = vids[i]
            target_data = self.data[self.data[:, 2] == target_vid]

            seq_len = target_data.shape[0]
            arr_idx = np.argwhere(target_data[:, 0] == curr_frm_idx)[0][0]

            # obs seq
            obs_start_idx = arr_idx - self.obs_len + 1
            obs_end_idx = arr_idx
            if (obs_start_idx < 0):
                obs_start_idx = 0
            obs_seq = np.full(shape=(self.obs_len, feat_dim), fill_value=np.nan)
            obs_seq[obs_start_idx-(obs_end_idx-self.obs_len+1):, :] = target_data[obs_start_idx:obs_end_idx + 1, :]

            # if (len(obs_seq.tolist()) > 0):
            #     num_padd = self.obs_len - obs_seq.shape[0]
            #     if (num_padd > 0):
            #         first_element = obs_seq[0, :].reshape(1, obs_seq.shape[1])
            #         for j in range(num_padd):
            #             obs_seq = np.concatenate([first_element, obs_seq], axis=0)
            # else:
            #     obs_seq = -1 * np.ones(shape=(self.obs_len, feat_dim))

            # pred seq
            pred_start_idx = arr_idx + 1
            pred_end_idx = arr_idx + self.pred_len
            if (pred_end_idx > seq_len - 1):
                pred_end_idx = seq_len - 1
            pred_seq = np.full(shape=(self.pred_len, feat_dim), fill_value=np.nan)
            pred_seq[:pred_end_idx - pred_start_idx + 1] = target_data[pred_start_idx:pred_end_idx + 1, :]

            # if (len(pred_seq.tolist()) > 0):
            #     num_padd = self.pred_len - pred_seq.shape[0]
            #     if (num_padd > 0):
            #         first_element = pred_seq[-1, :].reshape(1, pred_seq.shape[1])
            #         for j in range(num_padd):
            #             pred_seq = np.concatenate([pred_seq, first_element], axis=0)
            # else:
            #     pred_seq = -1 * np.ones(shape=(self.pred_len, feat_dim))

            trajs.append(np.expand_dims(np.concatenate([obs_seq, pred_seq], axis=0), axis=1))

        return np.concatenate(trajs, axis=1)

    def make_fake_label(self):

        '''
        ------------------------------------------------------------------------------------------------
        frame index| class | obj_id | x[m] | y[m] |  z[m]  | heading[rad] | width[m] | length[m] | height[m] | lidar | image
        ------------------------------------------------------------------------------------------------
        0          | 1     | 2      | 3    | 4    |   5    |      6       |    7     | 8         | 9         | 10     | 11
        ------------------------------------------------------------------------------------------------
        '''

        ego_data = self.data[self.data[:, 2] == -1]
        ego_cur = ego_data[0].reshape(1, ego_data.shape[1])

        offsets = 30.0 * 2 * (np.random.rand(self.num_fake_agents, 2) - 0.5)
        cur_frame = [ego_cur]
        for a in range(1, self.num_fake_agents):
            cur_agent = np.copy(ego_cur)
            cur_agent[0, 1] = 0
            cur_agent[0, 2] = a
            cur_agent[0, 3:5] += offsets[a-1, :]
            cur_agent[0, 7] = 2.06
            cur_agent[0, 8] = 4.59
            cur_agent[0, 9] = 1.54
            cur_frame.append(cur_agent)
        cur_frame = np.concatenate(cur_frame, axis=0)

        fake_data = [cur_frame]
        for f in range(1, 300):
            next_frame = np.copy(cur_frame)
            next_frame[:, 0] = f
            fake_data.append(next_frame)
        fake_data = np.concatenate(fake_data, axis=0)

        # save as csv file
        file_name = os.path.join(self.save_path, 'label_fake.csv')
        fp = open(file_name, 'w')
        csvWriter = csv.writer(fp, lineterminator='\n')
        csvWriter.writerow(
            ['frame index', 'class', 'obj_id', 'x[m]', 'y[m]', 'z[m]', 'heading[rad]', 'width[m]', 'length[m]',
             'height[m]', 'lidar file name', 'image file name'])

        for i in range(fake_data.shape[0]):
            cur_data = fake_data[i]
            cur_line = [str(int(cur_data[0]))]  # frame index
            cur_line.append(str(int(cur_data[1])))  # class
            cur_line.append(str(int(cur_data[2])))  # obj id
            cur_line.append(str(np.around(cur_data[3], decimals=5)))  # x
            cur_line.append(str(np.around(cur_data[4], decimals=5)))  # y
            cur_line.append(str(np.around(cur_data[5], decimals=5)))  # z
            cur_line.append(str(np.around(cur_data[6], decimals=5)))  # heading
            cur_line.append(str(np.around(cur_data[7], decimals=5)))  # w
            cur_line.append(str(np.around(cur_data[8], decimals=5)))  # l
            cur_line.append(str(np.around(cur_data[9], decimals=5)))  # h
            lidar_file_name = '%08d' % cur_data[10]
            image_file_name = '%08d' % cur_data[11]
            cur_line.append(lidar_file_name)  # lidar
            cur_line.append(image_file_name)  # image
            csvWriter.writerow(cur_line)
        fp.close()





def make_rot_matrix(heading, num_agents):

    m_cos = np.cos(heading).reshape(num_agents, 1)
    m_sin = np.sin(heading).reshape(num_agents, 1)
    m_R = np.concatenate([m_cos, -1 * m_sin, m_sin, m_cos], axis=1).reshape(num_agents, 2, 2)

    return m_R

def in_range_points(points, x, y, z, x_range, y_range, z_range):
    """ extract in-range points """
    points_select = points[np.logical_and.reduce((x > x_range[0], x < x_range[1], y > y_range[0], y < y_range[1], z > z_range[0], z < z_range[1]))]
    return np.around(points_select, decimals=2)

def example():

    parser = argparse.ArgumentParser()

    # Inputs
    parser.add_argument('--save_path', type=str, default='/home/dooseop/DATASET/voss/0001')
    parser.add_argument('--map_size', type=int, default=1280)
    parser.add_argument('--range_meter', type=int, default=80)
    parser.add_argument('--rotate_angle', type=float, default=0)
    parser.add_argument('--num_fake_agents', type=int, default=100)

    args = parser.parse_args()
    

    # ------------------------------------------
    # Prepare data and Define DataLoader
    # ------------------------------------------

    loader = ProcessedDataLoader(args.save_path, args.map_size, args.range_meter, num_fake_agents=args.num_fake_agents)

    loader.make_fake_label()

    # data_len = len(loader.frm_idx)
    # obs_len = 10
    # pred_len = 20
    #
    # for i in range(50, data_len - pred_len):
    #     cur_scene = loader.get_current_scene(i)
    #
    #     # show topview
    #     img = loader.draw_current_scene_topview(i, cur_scene, args.map_size, args.range_meter, args.rotate_angle, show_traj=True)
    #     cv2.imshow("top view", img)
    #     cv2.waitKey(0)
    #
    #     # show front
    #     img = loader.project_bbox3d_to_front_img(cur_scene)
    #     cv2.imshow("front", img)
    #     cv2.waitKey(0)
    #     print(" current frm idx : %d" % i)


if __name__ == '__main__':
    example()
