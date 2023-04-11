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

class Pose:

    def __init__(self, heading, position):
        '''
        heading (1) : radian
        position (1 x 3) : meter
        '''

        self.heading = heading
        self.position = position[:, :2].reshape(1, 2)  # global position
        self.xyz = position

        self.R_e2g = rotation_matrix(heading) # ego-centric to global coordinate
        self.R_g2e = np.linalg.inv(self.R_e2g) # global to ego-centric coordinate


    def to_agent(self, positions):
        '''
        Global to Agent Centric Coordinate System Conversion

        positions (N x 2)
        output (N x 2)
        '''

        trans = positions - self.position # seq_len x 2
        return np.matmul(self.R_g2e, trans.T).T

    def to_global(self, positions):
        '''
        Agent Centric to Global Coordinate System Conversion

        positions (N x 2)
        output (N x 2)
        '''

        return np.matmul(self.R_e2g, positions.T).T + self.position


class VossHelper:

    def __init__(self):

        '''
        '''


    def read_raw_file(self, file_dir):
        return np.genfromtxt(file_dir, delimiter=',')

    def get_pose(self, data, frm_idx, obj_id):
        '''
        ------------------------------------------------------------------------------------------------
        frame index| class | obj_id | x[m] | y[m] |  z[m]  | heading[rad] | width[m] | length[m] | height[m] | lidar | image
        ------------------------------------------------------------------------------------------------
        0          | 1     | 2      | 3    | 4    |   5    |      6       |    7     | 8         | 9         | 10     | 11
        ------------------------------------------------------------------------------------------------
        '''

        cur_frm_data = data[data[:, 0] == frm_idx]
        cur_ego_data = cur_frm_data[cur_frm_data[:, 2] == obj_id]

        heading = cur_ego_data[0, 6]
        position = cur_ego_data[0, 3:6].reshape(1, 3)

        return Pose(heading=heading, position=position)

    def get_label_object(self, data, frm_idx):

        cur_frm_data = data[data[:, 0] == frm_idx]
        obj_ids = np.unique(cur_frm_data[:, 2]).tolist()
        obj_ids.remove(-1)
        num_labels = len(obj_ids)

        if (num_labels == 0):
            return []

        obj_dicts = []
        for i, idx in enumerate(obj_ids):

            cur_obj_data = cur_frm_data[cur_frm_data[:, 2] == idx] # 1 x dim
            obj_class = self.code_to_object_class(cur_obj_data[0, 1])
            obj_dict = {'obj_id' : idx,
                        'obj_class' : obj_class,
                        'wlh' : [cur_obj_data[0, 7], cur_obj_data[0, 8], cur_obj_data[0, 9]]}
            obj_dicts.append(obj_dict)

        return obj_dicts

    # update, 220106
    def code_to_object_class(self, code):

        '''
        code 0 : vehicle
        code 1 : truck
        code 2 : pedestrian
        code 3 : cyclist
        '''

        if (code == 0 or code == 1):
            return 'vehicle'
        elif (code == 2 or code == 3):
            return 'pedestrian'
        else:
            return 'unknown'

    def get_sensor_file_names(self, data, frm_idx):
        cur_frm_data = data[data[:, 0] == frm_idx]
        return cur_frm_data[0, 10], cur_frm_data[0, 11]


def rotation_matrix(heading):

    m_cos = np.cos(heading)
    m_sin = np.sin(heading)
    m_R = np.array([m_cos, -1 * m_sin, m_sin, m_cos]).reshape(2, 2)
    return m_R

def in_range_points(points, x, y, z, x_range, y_range, z_range):
    """ extract in-range points """
    points_select = points[np.logical_and.reduce((x > x_range[0], x < x_range[1], y > y_range[0], y < y_range[1], z > z_range[0], z < z_range[1]))]
    return np.around(points_select, decimals=2)

