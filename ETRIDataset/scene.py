from utils.libraries import *

class Scene:

    def __init__(self, log_token=None, time_index=None, agent_dict=None, city_name=None):

        self.log_token = log_token
        self.agent_dict = agent_dict
        self.time_index = int(time_index)
        self.num_agents = len(agent_dict)
        self.city_name = city_name
        self.voxel_file_name = None
        self.lidar_file_name = None
        self.image_file_name = None

    def make_id_2_token_lookup(self):
        self.id_2_token_lookup = {}
        for idx, key in enumerate(self.agent_dict):
            self.id_2_token_lookup[self.agent_dict[key].agent_id] = key

    def __repr__(self):
        return f"Scene ID: {self.log_token}," \
               f" City: {self.city_name}," \
               f" Num agents: {self.num_agents}."


class AgentCentricScene:

    def __init__(self, log_token=None, time_index=None, agent_token=None, city_name=None):

        self.log_token = log_token
        self.agent_token = agent_token
        self.time_index = int(time_index)
        self.city_name = city_name

        self.trajectories = None
        self.bboxes = None
        self.R_a2g = None
        self.R_g2a = None
        self.trans_g = None
        self.agent_ids = None
        self.target_agent_index = None


    def __repr__(self):
        return f"Scene ID: {self.log_token}," \
               f" City: {self.city_name}," \
               f" Agent ID: {self.agent_token}."