import numpy as np

class Observation:
    def __init__(self, state):
        self.state = state
        self.dist = 0
        self.dist_prev = 0
        self.reward = 0.0
        self.is_done = False
        self.is_truncated = False
        self.info = {}
        self.sim_step_no = 0
        self.angle = 0
        
    def reset(self, state):
        self.state = state
        self.dist = 0
        self.dist_prev = 0
        self.reward = 0.0
        self.is_done = False
        self.is_truncated = False
        self.info = {}
        self.sim_step_no = 0
        self.angle = 0


    # def reset(self, state):
    #     self.state = state

    def cur_observation(self):
        return np.array(self.state).astype(np.float32), float(self.reward), bool(self.is_done), bool(self.is_truncated), self.info
    
    def dict_observation(self):
        return self.state, float(self.reward), bool(self.is_done), bool(self.is_truncated), self.info