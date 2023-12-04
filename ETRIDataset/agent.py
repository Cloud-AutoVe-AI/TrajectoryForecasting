from utils.functions import *
from utils.trajectory_filter import LinearModelIntp, AverageFilter, HoleFilling
from ETRIDataset.VossHelper import Pose

class Agent:

    def __init__(self, type=None, attribute=None, track_id=None, agent_id=None, obs_len=None, pred_len=None):

        '''
        type : category
        attribute : attribute
        track_id : annotation_token
        agent_id : agent index in a scene
        obs_len : num positions in a past trajectory
        pred_len : num positions in a future trajectory
        '''


        self.type = type
        self.attribute = attribute
        self.track_id = track_id
        self.agent_id = agent_id

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.trajectory = np.full(shape=(obs_len+pred_len, 4), fill_value=np.nan)
        self.trajectory_proc = np.full(shape=(obs_len + pred_len, 4), fill_value=np.nan)  # update, 211216

        self.bbox_g = None
        self.bbox_e = None
        self.wlh = None
        self.pose = None
        
        self.test = None
        self.testtest = None
        self.testtesttest = None
        self.testtestttets = None


    def bbox2D(self):
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

        # 2D bbox
        w, l, h = self.wlh
        corner_b0 = np.array([l / 2, w / 2]).reshape(1, 2)
        corner_b1 = np.array([-l / 2, w / 2]).reshape(1, 2)
        corner_b2 = np.array([-l / 2, -w / 2]).reshape(1, 2)
        corner_b3 = np.array([l / 2, -w / 2]).reshape(1, 2)
        bbox = np.concatenate([corner_b0, corner_b1, corner_b2, corner_b3], axis=0)  # 4 x 2

        # agent to global coord
        self.bbox_g = self.pose.to_global(bbox)


    def __repr__(self):
        return self.type + '/' + self.track_id
