import numpy as np
import gymnasium as gym

class CurriculumWrapper(gym.Wrapper):

    def generate_difficulty_settings(start_settings, end_settings, steps):
        difficulty_settings = []
        for step in range(steps + 1):
            trans_tolerance = np.linspace(start_settings['trans_tolerance'], end_settings['trans_tolerance'], steps + 1)[step]
            angle_tolerance = np.linspace(np.rad2deg(start_settings['angle_tolerance']), np.rad2deg(end_settings['angle_tolerance']), steps + 1)[step]
            random_range = np.array([np.linspace(start, end, steps + 1)[step] for start, end in zip(start_settings['random_range'], end_settings['random_range'])], dtype=np.float32)

            difficulty_settings.append({
                'trans_tolerance': trans_tolerance,
                'angle_tolerance': np.deg2rad(angle_tolerance),
                'random_range': random_range,
            })
        
        return difficulty_settings


    def __init__(self, env):
        super().__init__(env)
        start_settings = {
            'trans_tolerance': 0.004, 
            'angle_tolerance': np.deg2rad(10), 
            'random_range': np.array([0.004, 0.035, np.pi/4], dtype=np.float32), 
        }


        end_settings = {
            'trans_tolerance': 0.001, 
            'angle_tolerance': np.deg2rad(10), 
            'random_range': np.array([0.004, 0.035, np.pi/4], dtype=np.float32), 
        }

        steps = 10  
        difficulty_settings = CurriculumWrapper.generate_difficulty_settings(start_settings, end_settings, steps)

        self.difficulty_settings = difficulty_settings
        self.current_difficulty_level = 0
        self.successful_episodes = 0
        self.total_episodes = 0
        self.success_threshold = 0.9  # 90% success rate
        self.env.update_difficulty(self.difficulty_settings[self.current_difficulty_level])
 
        
    def reset(self, **kwargs):
        if self.total_episodes > 5 and self.successful_episodes / self.total_episodes > self.success_threshold:
            self.current_difficulty_level = min(self.current_difficulty_level + 1, len(self.difficulty_settings) - 1)
            self.env.update_difficulty(self.difficulty_settings[self.current_difficulty_level])
            self.successful_episodes = 0
            self.total_episodes = 0
        
        self.total_episodes += 1
        print(f"Current Curriculum: {self.current_difficulty_level}\n")

        if (self.total_episodes >= 20):
            self.successful_episodes = 0
            self.total_episodes = 0

        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, terminated, truncated, info  = self.env.step(action)
        if terminated:
            if 'is_success' in info and info['is_success']:
                self.successful_episodes += 1
        return obs, reward, terminated, truncated, info
