import gymnasium as gym
from stable_baselines3.common.monitor import Monitor


class GymEnv:
    def __init__(self, env_name, env_arguments):
        self.env = None
        self.env_name = env_name
        self.env_arguments = env_arguments

    def create_env(self):
        self.env = gym.make(self.env_name, **self.env_arguments)
        self.env = Monitor(self.env)

    def print_env_info(self):
        if self.env is not None:
            print("The name of Enviornment:", self.env_name)
            print("The action space:", self.env.action_space)
            print("The Observation space:", self.env.observation_space)
            #print("The reward range:", self.env.reward_range)
            print("The env spec:", self.env.spec)
            print("The metadata:", self.env.metadata)
        else:
            print("Enviornment is not created yet.")


if __name__ == "__main__":
    env_arguments = {
        'domain_randomize': False,
        'continuous': False,
        'render_mode': 'rgb_array'
    }
    env = GymEnv('CarRacing-v2', env_arguments)
    env.create_env()
    env.print_env_info()
