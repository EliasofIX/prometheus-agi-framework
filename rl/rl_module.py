# rl/rl_module.py
import gym
from stable_baselines3 import PPO

def create_generic_env():
    return gym.make("CartPole-v1")

class GenericRLAgent:
    """
    A generic RL agent for task-independent learning.
    """
    def __init__(self, env_fn=create_generic_env, model_path=None):
        self.env = env_fn()
        if model_path:
            self.model = PPO.load(model_path)
        else:
            self.model = PPO("MlpPolicy", self.env, verbose=0)
    
    def train(self, timesteps=10000):
        self.model.learn(total_timesteps=timesteps)
    
    def predict(self, observation):
        action, _ = self.model.predict(observation, deterministic=True)
        return action

