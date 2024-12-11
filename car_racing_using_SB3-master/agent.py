import os
import sys
from typing import List

import gymnasium
import imageio
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3 import TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

from wandb.integration.sb3 import WandbCallback

import wandb
from enviornment import GymEnv

sys.modules["gym"] = gymnasium


class SB3Agent:
    def __init__(self, agent_name: str, agent_algo: str, the_env: GymEnv, model_location: str):
        self.is_trained = False
        self.train_callback = None
        self.model = None
        self.agent_algorithm = agent_algo
        self.agent_policy = None
        self.agent_name = agent_name
        self.model_location = model_location
        self.model_env = the_env

    def create_new_agent(self, policy: str, algorithm_arguments):
        self.agent_policy = policy
        if self.agent_algorithm == 'DQN':
            self.model = DQN(self.agent_policy, self.model_env.env, **algorithm_arguments)
        print("Instantiation of Agent done..")

    def train_agent(self, train_arguments, callbacks: [List[BaseCallback]] = None):
        self.train_callback = callbacks
        self.model.learn(**train_arguments, callback=self.train_callback)
        self.is_trained = True
        print("Training complete..")

    def load_a_trained_agent(self):
        if self.agent_algorithm == 'DQN':
            self.model = DQN.load(self.model_location, self.model_env.env)
        self.is_trained = True
        print("Loading model complete..")

    def evaluate_agent(self, eval_episodes=20, callback=None, evaluate_args={}):
        mean_reward, std_reward = evaluate_policy(self.model, self.model_env.env, n_eval_episodes=eval_episodes,
                                                  callback=callback, **evaluate_args)
        print(f"The mean_reward across {eval_episodes}: {mean_reward}")
        print(f"The std_reward across {eval_episodes}: {std_reward}")

    def save_trained_agent(self, name: str):
        path = os.path.join(self.model_location, name)
        self.model.save(path)
        print("model saved in location:", path)

    def enjoy_trained_agent(self, num_steps=1000):
        vec_env = DummyVecEnv([lambda: self.model_env.env])
        obs = vec_env.reset()
        for i in range(num_steps):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, rewards, dones, info = vec_env.step(action)
            vec_env.render(mode='human')

    def record_agent(self, record_location: str, video_length: int, name_prefix: str):
        # It requires ffmpeg or avconv to be installed on the machine
        vec_env = DummyVecEnv([lambda: self.model_env.env])
        vec_env = VecVideoRecorder(vec_env, record_location,
                                   record_video_trigger=lambda x: x == 0, video_length=video_length,
                                   name_prefix=name_prefix)
        obs = vec_env.reset()
        for _ in range(video_length + 1):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, _, _ = vec_env.step(action)
        vec_env.close()

    def make_agent_gif(self, gif_location: str, gif_length: int):
        images = []
        vec_env = DummyVecEnv([lambda: self.model_env.env])
        obs = vec_env.reset()
        img = vec_env.render(mode='rgb_array')
        for i in range(gif_length):
            images.append(img)
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, _, _ = vec_env.step(action)
            img = vec_env.render(mode='rgb_array')
        imageio.mimsave(gif_location, [np.array(img) for i, img in enumerate(images) if i % 2 == 0], fps=29)


def get_env(env_arguments):
    env = GymEnv('CarRacing-v2', env_arguments)
    env.create_env()
    env.print_env_info()
    return env


def train_new_agent(agent_name, env, algorithm, policy, algo_arguments, train_arguments,
                    callbacks, save_location):
    run = wandb.init(
        project="stable_baselines3_experiments",
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,
    )
    vec_env = DummyVecEnv([lambda: env.env])
    vec_env = VecVideoRecorder(vec_env, f"videos/{run.id}", record_video_trigger=lambda x: x % 100000 == 0,
                               video_length=1000)
    env.env = vec_env
    agent = SB3Agent(agent_name, algorithm, env, save_location)
    agent.create_new_agent(policy, algo_arguments)
    agent.train_agent(train_arguments, callbacks)
    agent.save_trained_agent("DQN-newly-trained")
    return agent


def load_and_enjoy(agent: SB3Agent, eval_episodes=20, enjoy_steps=1000):
    agent.load_a_trained_agent()
    agent.evaluate_agent(eval_episodes=eval_episodes)
    agent.enjoy_trained_agent(num_steps=enjoy_steps)


if __name__ == "__main__":
    # create an enviornment
    env_arguments = {
        'domain_randomize': False,
        'continuous': False,
        'render_mode': 'rgb_array'
    }
    the_env = get_env(env_arguments)
    # create train and save agent
    agent_name = 'dqn-car-racing'
    save_location = './saved_models'
    algorithm_arguments = {'verbose': 1, 'buffer_size': 250000, 'learning_starts': 10000, 'batch_size': 32,
                           'tensorboard_log': "runs/second_run"}
    algorithm = 'DQN'
    policy = 'CnnPolicy'
    train_arguments = {
        'total_timesteps': 1e6,
        'log_interval': 5, 'progress_bar': True,
    }
    # the_callbacks = [WandbCallback(verbose=2)]
    # agent = train_new_agent(agent_name, the_env, algorithm, policy, algorithm_arguments, train_arguments, the_callbacks,
    #                         save_location)
    agent = SB3Agent('test-agent', 'DQN', the_env, './DQN-newly-trained')
    agent.load_a_trained_agent()
    # agent.enjoy_trained_agent()
    # agent.record_agent('./saved_models/recorded_videos', 2000, "DQN-car_racing_very_new")
    # agent.make_agent_gif('./saved_models/recorded_videos/dqn-car-racing-very-new.gif', 500)
    # evaluate trained agent

    agent.evaluate_agent(eval_episodes=20)
    # # record the trained agent's working
    # record_length = 5000
    # record_prefix = 'dqn-car-racing-new'
    # agent.record_agent(os.path.join(save_location, 'recorded_videos'), record_length, record_prefix)
