from utils.functions import *

class Dtype_Scene:

    def __init__(self, dataset=None, data_info=None, num_agents=None):

        self.dataset = dataset
        self.data_info = data_info
        self.loader_type = 'D'
        self.num_agents = num_agents

        self.obs_traj = None
        self.future_traj = None
        self.obs_traj_a = None
        self.future_traj_a = None
        self.feature_topview = None
        self.R_map = None
        self.R_traj = None
        self.agent_id = None

    def __repr__(self):
        return f" Dataset: {self.dataset}," \
               f" Loader Type: {self.loader_type}," \
               f" Num agents: {self.num_agents}."

