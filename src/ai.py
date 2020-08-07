import tensorflow as tf
import numpy as np
import gameAI

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

tf.compat.v1.enable_v2_behavior()

class SnakeEnv(py_environment.PyEnvironment):
    def __init__(self):
        self._action_spec = array_spec.BoundedArraySpec(
        shape=(), dtype=np.int32, minimum=0, maximum=2, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
        shape=(1,8), dtype=np.int32, minimum=[0]*8, name='observation')
        # Whether empty: Left, front, right
        # Whether food is in that direction: Left, front, right
        # Whether getting closer to food from last frame
        # Wheter eating food in current frame
        self._state = [0,0,0,0,0,0,0,0]
        self._episode_ended = False
    
    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = [0,0,0,0,0,0,0,0]
        self._episode_ended = False
        return ts.restart(np.array([self._state], dtype=np.int32))

    def _step(self, action):
        if self._episode_ended:
            return self.reset()
        self._state = gameAI.step(action)
        if gameAI.done == True:
            self._episode_ended = True
            return ts.termination(np.array([self._state], dtype=np.int32), -10)
        
        if self._state[7] == 1:
            return ts.transition(np.array([self._state], dtype=np.int32), reward=1, discount=1.0)
        #small reward for going to food
        if self._state[6] == 1:
            return ts.transition(np.array([self._state], dtype=np.int32), reward=0.1, discount=1.0)
        #negative reward for going away from food
        else:
            return ts.transition(np.array([self._state], dtype=np.int32), reward=-0.15, discount=1.0)

if __name__ == '__main__':
    py_env = SnakeEnv()
    utils.validate_py_environment(py_env, episodes = 5)
    tf_env = tf_py_environment.TFPyEnvironment(py_env)
