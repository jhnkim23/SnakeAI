from __future__ import absolute_import, division, print_function
from gameAI import Snake
import time
# import abc
# import tensorflow as tf
# import numpy as np

#Environment
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.environments import wrappers

# #Training
# from tf_agents.agents.dqn import dqn_agent
# from tf_agents.networks import q_network
# from tf_agents.drivers import dynamic_step_driver
# from tf_agents.environments import tf_py_environment
# from tf_agents.trajectories import trajectory
# from tf_agents.environments import wrappers
# #from tf_agents.metrics import metric_utils
# from tf_agents.metrics import tf_metrics
# from tf_agents.policies import random_tf_policy
# from tf_agents.replay_buffers import tf_uniform_replay_buffer
# from tf_agents.utils import common
# from tf_agents.metrics import py_metrics
# from tf_agents.metrics import tf_metrics
# from tf_agents.drivers import py_driver
# from tf_agents.drivers import dynamic_episode_driver

import base64
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
# from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

# import matplotlib.pyplot as plt

# tf.compat.v1.enable_v2_behavior()

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
        self.game = Snake()
        self.reward_total = 0
    
    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = [0,0,0,0,0,0,0,0]
        self._episode_ended = False
        self.reward_total = 0
        self.game = Snake()
        return ts.restart(np.array([self._state], dtype=np.int32))

    def _step(self, action):
        time.sleep(.001)
        if self._episode_ended:
            return self.reset() 
        self._state = self.game.step(action - 1)
        if self.game.done == True or self.reward_total < -10:
            self._episode_ended = True
            self.reward_total += -10
            return ts.termination(np.array([self._state], dtype=np.int32), -10)
        
        if self._state[7] == 1:
            self.reward_total += 1
            return ts.transition(np.array([self._state], dtype=np.int32), reward=1, discount=1.0)
        #small reward for going to food
        if self._state[6] == 1:
            self.reward_total += 0.1
            return ts.transition(np.array([self._state], dtype=np.int32), reward=0.1, discount=1.0)
        #negative reward for going away from food
        else:
            self.reward_total += -0.15
            return ts.transition(np.array([self._state], dtype=np.int32), reward=-0.15, discount=1.0)

if __name__ == '__main__':
    # Environment
    env = tf_py_environment.TFPyEnvironment(SnakeEnv())

    # Q net
    q_net = q_network.QNetwork(env.observation_spec(), env.action_spec())

    # Optimizer
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)

    # Agent
    train_step_counter = tf.compat.v2.Variable(0)

    agent = dqn_agent.DqnAgent(env.time_step_spec(),
                            env.action_spec(),
                            q_network=q_net,
                            optimizer=optimizer,
                            td_errors_loss_fn=common.element_wise_squared_loss,
                            train_step_counter=train_step_counter)

    agent.initialize()

    # Conpute average return
    def compute_avg_return(environment, policy, num_episodes=10):
        total_return = 0.0
        for _ in range(num_episodes):

            time_step = environment.reset()
            episode_return = 0.0

            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = environment.step(action_step.action)
                episode_return += time_step.reward
            total_return += episode_return

        avg_return = total_return / num_episodes
        return avg_return.numpy()[0]

    # Evaluate the agent's policy once before training.
    avg_return = compute_avg_return(env, agent.policy, 5)
    returns = [avg_return]

    #Buffer
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec=agent.collect_data_spec,
                                                                batch_size=env.batch_size,
                                                                max_length=100000)

    def collect_step(environment, policy, buffer):
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        # Add trajectory to the replay buffer
        buffer.add_batch(traj)
    
    #Train
    collect_steps_per_iteration = 1
    batch_size = 64
    dataset = replay_buffer.as_dataset(num_parallel_calls=3, 
                                        sample_batch_size=batch_size, 
                                        num_steps=2).prefetch(3)
    iterator = iter(dataset)
    num_iterations = 20000
    env.reset()

    for _ in range(batch_size):
        collect_step(env, agent.policy, replay_buffer)

    for _ in range(num_iterations):
        # Collect a few steps using collect_policy and save to the replay buffer.
        for _ in range(collect_steps_per_iteration):
            collect_step(env, agent.collect_policy, replay_buffer)

        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss

        step = agent.train_step_counter.numpy()

        # Print loss every 200 steps.
        if step % 200 == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss))

        # Evaluate agent's performance every 1000 steps.
        if step % 1000 == 0:
            avg_return = compute_avg_return(env, agent.policy, 5)
            print('step = {0}: Average Return = {1}'.format(step, avg_return))
            returns.append(avg_return)

    #Plot
    plt.figure(figsize=(12,8))
    iterations = range(0, num_iterations + 1, 1000)
    plt.plot(iterations, returns)
    plt.ylabel('Average Return')
    plt.xlabel('Iterations')
    plt.show()

# if __name__ == '__main__':
#     #Environments
#     train_py_env = SnakeEnv()
#     eval_py_env = SnakeEnv()

#     train_env = tf_py_environment.TFPyEnvironment(train_py_env)
#     eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

#     #Parameters
#     num_iterations = 10000  # @param

#     initial_collect_steps = 1000  # @param
#     collect_steps_per_iteration = 1  # @param
#     replay_buffer_capacity = 100000  # @param

#     fc_layer_params = (100, 50, 25,)

#     batch_size = 128  # @param
#     learning_rate = 1e-5  # @param
#     log_interval = 200  # @param

#     num_eval_episodes = 2  # @param
#     eval_interval = 1000  # @param

#     #Agent
    
#     q_net = q_network.QNetwork(
#         train_env.observation_spec(),
#         train_env.action_spec(),
#         fc_layer_params=fc_layer_params)

#     optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

#     train_step_counter = tf.compat.v2.Variable(0)

#     tf_agent = dqn_agent.DqnAgent(
#             train_env.time_step_spec(),
#             train_env.action_spec(),
#             q_network=q_net,
#             optimizer=optimizer,
#             # td_errors_loss_fn = dqn_agent.element_wise_squared_loss,
#             train_step_counter=train_step_counter)

#     tf_agent.initialize()

#     #Policy
#     eval_policy = tf_agent.policy
#     collect_policy = tf_agent.collect_policy

#     #Replay buffer and observer
#     replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
#         data_spec=tf_agent.collect_data_spec,
#         batch_size=train_env.batch_size,
#         max_length=replay_buffer_capacity)

#     replay_observer = [replay_buffer.add_batch]
    
#     #Dataset
#     dataset = replay_buffer.as_dataset(
#             num_parallel_calls=3,
#             sample_batch_size=batch_size,
#         num_steps=3).prefetch(3)
    
#     iterator = iter(dataset)

#     #Driver
#     train_metrics = [
#         tf_metrics.NumberOfEpisodes(),
#         tf_metrics.EnvironmentSteps(),
#         tf_metrics.AverageReturnMetric(),
#         tf_metrics.AverageEpisodeLengthMetric(),
#     ]

#     driver = dynamic_step_driver.DynamicStepDriver(
#             train_env,
#             collect_policy,
#             observers=replay_observer + train_metrics,
#         num_steps=1)

#     #Training
#     episode_len = []

#     final_time_step, policy_state = driver.run()

#     for i in range(num_iterations):
#         final_time_step, _ = driver.run(final_time_step, policy_state)

#         experience, _ = next(iterator)
#         train_loss = tf_agent.train(experience=experience)
#         step = tf_agent.train_step_counter.numpy()

#         if step % log_interval == 0:
#             print('step = {0}: loss = {1}'.format(step, train_loss.loss))
#             episode_len.append(train_metrics[3].result().numpy())
#             print('Average episode length: {}'.format(train_metrics[3].result().numpy()))

#         if step % eval_interval == 0:
#             avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
#             print('step = {0}: Average Return = {1}'.format(step, avg_return))
#     plt.plot(episode_len)
#     plt.show()