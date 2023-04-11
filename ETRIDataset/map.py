import cv2
import numpy as np

from utils.functions import *
from shapely import affinity
import colorsys
from shapely.geometry import MultiPolygon
from random import randint # update, 220216

class Map:

    def __init__(self, args):

        self.root_path = './ETRIDataset/HDmap'
        file_name_laneside = 'etridb_plus_LAYER_LANESIDE.pkl'
        file_name_lanelink = 'etridb_plus_LAYER_LN_LINK.pkl'
        file_name_lanenode = 'etridb_plus_LAYER_LN_NODE.pkl'
        file_name_roadmark = 'etridb_plus_LAYER_ROADMARK.pkl'

        rt = self.load_laneside(os.path.join(self.root_path, file_name_laneside))
        rt = self.load_lanelink(os.path.join(self.root_path, file_name_lanelink))
        rt = self.load_lanenode(os.path.join(self.root_path, file_name_lanenode))
        rt = self.load_roadmark(os.path.join(self.root_path, file_name_roadmark))

        try:
            self.is_shift_lane_deg = args.is_shift_lane_deg
        except:
            self.is_shift_lane_deg = 0

        try:
            self.hdmap_type = args.hdmap_type
        except:
            self.hdmap_type = 0

        # update, 220516
        try:
            self.lane_color_prob = args.lane_color_prob
        except:
            self.lane_color_prob = 0


    def return_centerlines(self, pose, x_range, y_range):

        xy = pose.position[0]
        w_x_min = xy[0] + x_range[0] - 10
        w_x_max = xy[0] + x_range[1] + 10
        w_y_min = xy[1] + y_range[0] - 10
        w_y_max = xy[1] + y_range[1] + 10
        win_min_max = (w_x_min, w_y_min, w_x_max, w_y_max)

        lane_segs, IDs = [], []
        for idx, key in enumerate(self.lanelink):

            lane_seg = self.lanelink[key]
            ID = lane_seg['ID']
            lane_min_max = (lane_seg['Cover'][0], lane_seg['Cover'][1], lane_seg['Cover'][2], lane_seg['Cover'][3])

            if (correspondance_check(win_min_max, lane_min_max) == True):
                positions = lane_seg['Pts'] # N x 2
                positions = pose.to_agent(positions) # ego-centric
                lane_segs.append(positions)
                IDs.append(ID)

        return lane_segs, IDs

    def return_laneside(self, pose, x_range, y_range):

        xy = pose.position[0]
        w_x_min = xy[0] + x_range[0] - 10
        w_x_max = xy[0] + x_range[1] + 10
        w_y_min = xy[1] + y_range[0] - 10
        w_y_max = xy[1] + y_range[1] + 10
        win_min_max = (w_x_min, w_y_min, w_x_max, w_y_max)

        lane_segs = []
        for idx, key in enumerate(self.laneside):

            lane_seg = self.laneside[key]
            lane_min_max = (lane_seg['Cover'][0], lane_seg['Cover'][1], lane_seg['Cover'][2], lane_seg['Cover'][3])

            if (correspondance_check(win_min_max, lane_min_max) == True):
                positions = lane_seg['Pts'] # N x 2
                positions = pose.to_agent(positions) # ego-centric
                lane_segs.append(positions)

        return lane_segs

    def return_roadmark(self, pose, x_range, y_range):

        xy = pose.position[0]
        w_x_min = xy[0] + x_range[0] - 10
        w_x_max = xy[0] + x_range[1] + 10
        w_y_min = xy[1] + y_range[0] - 10
        w_y_max = xy[1] + y_range[1] + 10
        win_min_max = (w_x_min, w_y_min, w_x_max, w_y_max)

        poly_segs = []
        for idx, key in enumerate(self.roadmark):

            lane_seg = self.roadmark[key]
            lane_min_max = (lane_seg['Cover'][0], lane_seg['Cover'][1], lane_seg['Cover'][2], lane_seg['Cover'][3])

            if (correspondance_check(win_min_max, lane_min_max) == True):
                positions = lane_seg['Pts'] # N x 2
                positions = pose.to_agent(positions) # ego-centric
                poly_segs.append(positions)

        return poly_segs

    def make_topview_map_TypeD(self, pose, x_range, y_range, map_size, obs_traj, category, isTrain=False):

        '''
        pose : class-type
        x_range = (min, max)
        y_range = (min, max)
        map_size
        obs_pos : seq_len x batch x 2
        category : 1 x seq_len

        output : map_size x map_size x 4 (0~1)
        '''

        # color type
        if (self.hdmap_type == 0):
            img = np.zeros(shape=(map_size, map_size, 3))

            # map_size x map_size x 3, 0 ~ 255
            img = self.draw_roadmark(pose, x_range, y_range, map_size, brightness=64, img=img)

            # map_size x map_size x 3, 0 ~ 255
            img = self.draw_centerlines_color(img, pose, x_range, y_range, map_size, isTrain=isTrain)

            # cv2.imshow("", img.astype('uint8'))
            # cv2.waitKey(0)

            # map_size x map_size x 1, 0 ~ 255
            img_traj = self.draw_agent_trajectories(obs_traj, x_range, y_range, map_size, category)[:, :, 0].reshape(
                map_size, map_size, 1)

            # map_size x map_size x 4, 0 ~ 255
            img = np.concatenate([img, img_traj], axis=2)

        # multi-ch type
        else:

            # map_size x map_size x 1, 0 ~ 255
            img_roadmark = self.draw_roadmark(pose, x_range, y_range, map_size, brightness=255)

            # map_size x map_size x 4, 0 ~ 255
            img_centerline = self.draw_centerlines_multich(pose, x_range, y_range, map_size, isTrain)

            # map_size x map_size x 1, 0 ~ 255
            img_traj = self.draw_agent_trajectories(obs_traj, x_range, y_range, map_size, category)[:, :, 0].reshape(map_size, map_size, 1)

            # map_size x map_size x 6, 0 ~ 255
            img = np.concatenate([img_roadmark, img_centerline, img_traj], axis=2)

        # cv2.imshow("", img.astype('uint8'))
        # cv2.waitKey(0)

        return img.astype('uint8').astype('float') / 255.0


    def make_topview_map_typeE(self, pose, x_range, y_range, map_size, obs_traj, category):

        '''
        pose : class-type
        x_range = (min, max)
        y_range = (min, max)
        map_size
        obs_pos : seq_len x batch x 2
        category : 1 x seq_len

        output : map_size x map_size x 4 (0~1)
        '''

        img = np.zeros(shape=(map_size, map_size, 3))

        # map_size x map_size x 3, 0 ~ 255
        img = self.draw_roadmark(pose, x_range, y_range, map_size, brightness=64, img=img)

        # map_size x map_size x 3, 0 ~ 255
        img = self.draw_centerlines_color(img, pose, x_range, y_range, map_size)

        # map_size x map_size x 1, 0 ~ 255
        img_traj = self.draw_agent_trajectories(obs_traj, x_range, y_range, map_size, category)[:, :, 0].reshape(map_size, map_size, 1)

        # map_size x map_size x 4, 0 ~ 255
        img = np.concatenate([img, img_traj], axis=2)

        # cv2.imshow("", img.astype('uint8'))
        # cv2.waitKey(0)

        return img.astype('uint8').astype('float') / 255.0

    def draw_agent_trajectories(self, obs_traj, x_range, y_range, map_size, category):

        img = np.zeros(shape=(map_size, map_size, 3))

        seq_len, batch, dim = obs_traj.shape
        axis_range_y = y_range[1] - y_range[0]
        axis_range_x = x_range[1] - x_range[0]
        scale_y = float(map_size - 1) / axis_range_y
        scale_x = float(map_size - 1) / axis_range_x

        for b in range(batch):

            # vehicle 0, ped 1
            if (category[0, b] == 0):
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

    def draw_centerlines_multich(self, pose, x_range, y_range, map_size, isTrain=False):

        img_0 = np.zeros(shape=(map_size, map_size, 1))
        img_1 = np.zeros(shape=(map_size, map_size, 1))
        img_2 = np.zeros(shape=(map_size, map_size, 1))
        img_3 = np.zeros(shape=(map_size, map_size, 1))

        xy = pose.position[0]
        w_x_min = xy[0] + x_range[0] - 10
        w_x_max = xy[0] + x_range[1] + 10
        w_y_min = xy[1] + y_range[0] - 10
        w_y_max = xy[1] + y_range[1] + 10
        win_min_max = (w_x_min, w_y_min, w_x_max, w_y_max)

        axis_range_y = y_range[1] - y_range[0]
        axis_range_x = x_range[1] - x_range[0]
        scale_y = float(map_size - 1) / axis_range_y
        scale_x = float(map_size - 1) / axis_range_x

        for idx, key in enumerate(self.lanelink):

            lane_seg = self.lanelink[key]
            lane_ID = lane_seg['ID']
            lane_min_max = (lane_seg['Cover'][0], lane_seg['Cover'][1], lane_seg['Cover'][2], lane_seg['Cover'][3])

            if (correspondance_check(win_min_max, lane_min_max) == True):

                positions = lane_seg['Pts'] # N x 2
                positions = pose.to_agent(positions) # ego-centric

                diff = positions[1:] - positions[:-1]
                line_yaws = calc_yaw_from_points(diff)
                chk = line_yaws < 0
                line_yaws[chk] += 2*np.pi # 0 to 2*pi

                line_angle_deg = line_yaws * 180 / np.pi
                line_angle_deg = line_angle_deg % 360

                col_pels = -(positions[:, 1] * scale_y).astype(np.int32)
                row_pels = -(positions[:, 0] * scale_x).astype(np.int32)

                col_pels += int(np.trunc(y_range[1] * scale_y))
                row_pels += int(np.trunc(x_range[1] * scale_x))

                for j in range(1, positions.shape[0]):
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

        return np.concatenate([img_0, img_1, img_2, img_3], axis=2)

    def draw_centerlines_color(self, img, pose, x_range, y_range, map_size, isTrain=False):

        deg_shift = 0
        if (isTrain and self.is_shift_lane_deg == 1 and np.random.rand(1) < self.lane_color_prob):
            deg_shift = float(randint(0, 180))

        xy = pose.position[0]
        w_x_min = xy[0] + x_range[0] - 10
        w_x_max = xy[0] + x_range[1] + 10
        w_y_min = xy[1] + y_range[0] - 10
        w_y_max = xy[1] + y_range[1] + 10
        win_min_max = (w_x_min, w_y_min, w_x_max, w_y_max)

        axis_range_y = y_range[1] - y_range[0]
        axis_range_x = x_range[1] - x_range[0]
        scale_y = float(map_size - 1) / axis_range_y
        scale_x = float(map_size - 1) / axis_range_x

        for idx, key in enumerate(self.lanelink):

            lane_seg = self.lanelink[key]
            lane_ID = lane_seg['ID']
            lane_min_max = (lane_seg['Cover'][0], lane_seg['Cover'][1], lane_seg['Cover'][2], lane_seg['Cover'][3])

            if (correspondance_check(win_min_max, lane_min_max) == True):

                positions = lane_seg['Pts'] # N x 2
                positions = pose.to_agent(positions) # ego-centric

                diff = positions[1:] - positions[:-1]
                line_yaws = calc_yaw_from_points(diff) + np.pi  # 0 to 2*pi
                line_angle_deg = line_yaws * 180 / np.pi

                line_angle_deg += deg_shift
                line_angle_deg = line_angle_deg % 360

                col_pels = -(positions[:, 1] * scale_y).astype(np.int32)
                row_pels = -(positions[:, 0] * scale_x).astype(np.int32)

                col_pels += int(np.trunc(y_range[1] * scale_y))
                row_pels += int(np.trunc(x_range[1] * scale_x))

                for j in range(1, positions.shape[0]):
                    rgb_n = colorsys.hsv_to_rgb(line_angle_deg[j - 1] / 360, 1., 1.)
                    color = (int(255 * rgb_n[0]), int(255 * rgb_n[1]), int(255 * rgb_n[2]))

                    start = (col_pels[j], row_pels[j])
                    end = (col_pels[j-1], row_pels[j-1])
                    cv2.line(img, start, end, color, 1)

        return img

    def draw_laneside(self, img, pose, x_range, y_range, map_size):

        # img = np.zeros(shape=(map_size, map_size, 3))

        xy = pose.position[0]
        w_x_min = xy[0] + x_range[0] - 10
        w_x_max = xy[0] + x_range[1] + 10
        w_y_min = xy[1] + y_range[0] - 10
        w_y_max = xy[1] + y_range[1] + 10
        win_min_max = (w_x_min, w_y_min, w_x_max, w_y_max)

        axis_range_y = y_range[1] - y_range[0]
        axis_range_x = x_range[1] - x_range[0]
        scale_y = float(map_size - 1) / axis_range_y
        scale_x = float(map_size - 1) / axis_range_x

        for idx, key in enumerate(self.laneside):

            lane_seg = self.laneside[key]
            lane_min_max = (lane_seg['Cover'][0], lane_seg['Cover'][1], lane_seg['Cover'][2], lane_seg['Cover'][3])

            if (correspondance_check(win_min_max, lane_min_max) == True):
                positions = lane_seg['Pts'] # N x 2
                positions = pose.to_agent(positions) # ego-centric

                col_pels = -(positions[:, 1] * scale_y).astype(np.int32)
                row_pels = -(positions[:, 0] * scale_x).astype(np.int32)

                col_pels += int(np.trunc(y_range[1] * scale_y))
                row_pels += int(np.trunc(x_range[1] * scale_x))

                for j in range(1, positions.shape[0]):
                    start = (col_pels[j], row_pels[j])
                    end = (col_pels[j-1], row_pels[j-1])
                    cv2.line(img, start, end, (128, 128, 128), 1)

        return img

    def draw_roadmark(self, pose, x_range, y_range, map_size, brightness, img=None):

        if (img is None):
            img = np.zeros(shape=(map_size, map_size, 1))

        xy = pose.position[0]
        w_x_min = xy[0] + x_range[0] - 10
        w_x_max = xy[0] + x_range[1] + 10
        w_y_min = xy[1] + y_range[0] - 10
        w_y_max = xy[1] + y_range[1] + 10
        win_min_max = (w_x_min, w_y_min, w_x_max, w_y_max)

        axis_range_y = y_range[1] - y_range[0]
        axis_range_x = x_range[1] - x_range[0]
        scale_y = float(map_size - 1) / axis_range_y
        scale_x = float(map_size - 1) / axis_range_x

        for idx, key in enumerate(self.roadmark):

            lane_seg = self.roadmark[key]
            lane_min_max = (lane_seg['Cover'][0], lane_seg['Cover'][1], lane_seg['Cover'][2], lane_seg['Cover'][3])

            if (correspondance_check(win_min_max, lane_min_max) == True):
                positions = lane_seg['Pts'] # N x 2
                positions = pose.to_agent(positions) # ego-centric

                col_pels = -(positions[:, 1] * scale_y).astype(np.int32)
                row_pels = -(positions[:, 0] * scale_x).astype(np.int32)

                col_pels += int(np.trunc(y_range[1] * scale_y))
                row_pels += int(np.trunc(x_range[1] * scale_x))

                final_pts = np.concatenate([col_pels.reshape(positions.shape[0],1), row_pels.reshape(positions.shape[0],1)], axis=1)
                img = cv2.fillConvexPoly(img, final_pts, (brightness, brightness, brightness))

        return img

    def load_laneside(self, file_dir):

        # check if exists
        if (os.path.exists(file_dir)):
            with open(file_dir, 'rb') as f:
                self.laneside = pickle.load(f)
            return True

        # make if do not exist
        file = open(file_dir.replace('pkl', 'txt'), 'r')
        Lines = file.readlines()

        self.laneside = {}
        for line in Lines:
            elements = line.split()
            elements = np.array(elements).astype('float')
            num_elmts = len(elements)

            header = elements[:6].astype('int')
            positions = elements[6:].reshape(int((num_elmts - 6) / 2), 2)

            x_max = np.max(positions[:, 0])
            x_min = np.min(positions[:, 0])
            y_max = np.max(positions[:, 1])
            y_min = np.min(positions[:, 1])

            lane_seg = {'ID': header[0],
                        'MID': header[1],
                        'LaneID': header[2],
                        'Type': header[3],
                        'Color': header[4],
                        'NumPts': header[5],
                        'Pts': positions,
                        'Cover': [x_max, x_min, y_max, y_min]
                        }

            self.laneside[str(header[0])] = lane_seg

        with open(file_dir.replace('txt', 'pkl'), 'wb') as f:
            pickle.dump(self.laneside, f)

        return True

    def load_lanelink(self, file_dir):

        if (os.path.exists(file_dir)):
            with open(file_dir, 'rb') as f:
                self.lanelink = pickle.load(f)
            return True


        file = open(file_dir.replace('pkl', 'txt'), 'r')
        Lines = file.readlines()

        self.lanelink = {}
        for line in Lines:
            elements = line.split()
            elements = np.array(elements).astype('float')
            num_elmts = len(elements)

            header = elements[:21].astype('int')
            positions = elements[21:].reshape(int((num_elmts - 21) / 2), 2)

            x_max = np.max(positions[:, 0])
            x_min = np.min(positions[:, 0])
            y_max = np.max(positions[:, 1])
            y_min = np.min(positions[:, 1])

            lane_seg = {'ID': header[0],
                        'MID': header[1],
                        'LID': header[2],
                        'RID': header[3],
                        'InMID': header[4],
                        'InLID': header[5],
                        'InRID': header[6],
                        'OutMID': header[7],
                        'OutLID': header[8],
                        'OutRID': header[9],
                        'Junction': header[10],
                        'Type': header[11],
                        'SubType': header[12],
                        'Twoway': header[13],
                        'RLID': header[14],
                        'LLinkID': header[15],
                        'RLinkID': header[16],
                        'SNodeID': header[17],
                        'ENodeID': header[18],
                        'Speed': header[19],
                        'NumPts': header[20],
                        'Pts': positions,
                        'Cover': [x_max, x_min, y_max, y_min]
                        }

            self.lanelink[str(header[0])] = lane_seg

        with open(file_dir.replace('txt', 'pkl'), 'wb') as f:
            pickle.dump(self.lanelink, f)

        return True

    def load_lanenode(self, file_dir):

        # check if exists
        if (os.path.exists(file_dir)):
            with open(file_dir, 'rb') as f:
                self.lanenode = pickle.load(f)
            return True

        # make if do not exist
        file = open(file_dir.replace('pkl', 'txt'), 'r')
        Lines = file.readlines()

        self.lanenode = {}
        for line in Lines:
            elements = line.split()
            elements = np.array(elements).astype('float')
            num_elmts = len(elements)

            header = elements[:num_elmts-2].astype('int')
            positions = elements[num_elmts-2:].reshape(1, 2)



            lane_seg = {'ID': header[0],
                        'NumConLink': header[1],
                        'LinkID': header[2:],
                        'Pts': positions
                        }

            self.lanenode[str(header[0])] = lane_seg

        with open(file_dir.replace('txt', 'pkl'), 'wb') as f:
            pickle.dump(self.lanenode, f)

        return True

    def load_roadmark(self, file_dir):

        # check if exists
        if (os.path.exists(file_dir)):
            with open(file_dir, 'rb') as f:
                self.roadmark = pickle.load(f)
            return True

        # make if do not exist
        file = open(file_dir.replace('pkl', 'txt'), 'r')
        Lines = file.readlines()

        self.roadmark = {}
        for line in Lines:
            elements = line.split()
            elements = np.array(elements).astype('float')
            num_elmts = len(elements)

            ID = elements[0].astype('int')
            Type = elements[1].astype('int')
            SubType = elements[2].astype('int')
            NumStopLine = elements[3].astype('int')
            if (NumStopLine > 0):
                StoplineID = elements[4:4+NumStopLine].astype('int')
                NumPts = elements[4+NumStopLine].astype('int')
                positions = elements[4+NumStopLine+1:].reshape(int((num_elmts - (4+NumStopLine)) / 2), 2)
            else:
                StoplineID = -1
                NumPts = elements[4].astype('int')
                positions = elements[5:].reshape(int((num_elmts - 4) / 2), 2)

            x_max = np.max(positions[:, 0])
            x_min = np.min(positions[:, 0])
            y_max = np.max(positions[:, 1])
            y_min = np.min(positions[:, 1])

            lane_seg = {'ID': ID,
                        'Type': Type,
                        'SubType': SubType,
                        'NumStopLine': NumStopLine,
                        'StoplineID': StoplineID,
                        'NumPts': NumPts,
                        'Pts': positions,
                        'Cover': [x_max, x_min, y_max, y_min]
                        }

            self.roadmark[str(ID)] = lane_seg

        with open(file_dir.replace('txt', 'pkl'), 'wb') as f:
            pickle.dump(self.roadmark, f)

        return True


    def __repr__(self):
        return f"ETRI Map Helper."




def in_range_points(points, x, y, z, x_range, y_range, z_range):

    points_select = points[np.logical_and.reduce((x > x_range[0], x < x_range[1], y > y_range[0], y < y_range[1], z > z_range[0], z < z_range[1]))]
    return np.around(points_select, decimals=2)

def correspondance_check(win_min_max, lane_min_max):

    # four points for window and lane box
    w_x_min, w_y_min, w_x_max, w_y_max = win_min_max
    # l_x_min, l_y_min, l_x_max, l_y_max = lane_min_max
    l_x_max, l_x_min, l_y_max, l_y_min = lane_min_max

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
    x1 = vec1[:, 0]
    y1 = vec1[:, 1]
    heading = np.arctan2(y1, x1)

    return heading
