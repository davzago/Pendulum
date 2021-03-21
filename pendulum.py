import gym
import numpy as np
import tensorflow as tf
import imageio


from tf_agents.specs import BoundedArraySpec
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.environments import wrappers
from tf_agents.specs import tensor_spec
from tf_agents.networks import sequential
from tf_agents.agents.dqn import dqn_agent
from tf_agents.utils import common
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tensorflow.data import DatasetSpec
from tf_agents.drivers import dynamic_step_driver
from tf_agents.trajectories.time_step import TimeStep


num_iterations = 8000 # @param {type:"integer"}

initial_collect_steps = 100  # @param {type:"integer"} 
collect_steps_per_iteration = 1  # @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}

batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
log_interval = 200  # @param {type:"integer"}

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}

env_name = 'Pendulum-v0'
num_actions = 5

"""envlol = gym.make(env_name)
dims = envlol.action_space.shape[0]
print(dims)"""

"""env = wrappers.ActionDiscretizeWrapper(suite_gym.load(env_name), num_actions=9)
#print('Discretized Action Spec:', env.action_spec())
env.reset()
observations = []
rewards = []
done = []
infos = []
initial_state = env.reset()
observations.append(initial_state)
action = [8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,4,4,4,4,4] #0,1,2,3,4,5,6,7,8
for i in range(20):
    env.render() 
    #action = 0 # env.action_space.sample()
    (obs, reward, d, info) = env.step(action[i]) # take a random action
    observations.append(obs)
    rewards.append(reward)
    done.append(d)
    infos.append(info)

time_step = env.reset()"""

train_py_env = wrappers.ActionDiscretizeWrapper(suite_gym.load(env_name), num_actions=num_actions)
eval_py_env = wrappers.ActionDiscretizeWrapper(suite_gym.load(env_name), num_actions=num_actions)
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
#print("action space:", env.action_spec())

"observation specification: BoundedArraySpec(shape=(3,), dtype=dtype('float32'), name='observation', minimum=[-1. -1. -8.], maximum=[1. 1. 8.])"
"reward specification: ArraySpec(shape=(), dtype=dtype('float32'), name='reward')"

fc_layer_params = (100, 50)
#print("Num actions:", num_actions)

def dense_layer(num_units):
  return tf.keras.layers.Dense(
      num_units,
      activation=tf.keras.activations.relu,
      kernel_initializer=tf.keras.initializers.VarianceScaling(
          scale=2.0, mode='fan_in', distribution='truncated_normal'))

dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
q_values_layer = tf.keras.layers.Dense(
    num_actions,
    activation=None,
    kernel_initializer=tf.keras.initializers.RandomUniform(
        minval=-0.03, maxval=0.03),
    bias_initializer=tf.keras.initializers.Constant(-0.2))
q_net = sequential.Sequential(dense_layers + [q_values_layer])

#print(q_net.summary())

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

train_step_counter = tf.Variable(0)

agent = dqn_agent.DdqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter,
    target_update_period=20,
    gamma=0.9)

agent.initialize()

eval_policy = agent.policy
collect_policy = agent.collect_policy

eval_policy = agent.policy
collect_policy = agent.collect_policy
random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())

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

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec = agent.collect_data_spec,
    batch_size = train_env.batch_size,
    max_length = replay_buffer_max_length 
)
print("train_env.batch_size:", train_env.batch_size)

replay_observer = [replay_buffer.add_batch]

train_env.reset()
collect_op = dynamic_step_driver.DynamicStepDriver(
  train_env,
  agent.collect_policy,
  observers=replay_observer,
  num_steps=initial_collect_steps)

"""def collect_step(environment, policy, buffer):
  time_step = environment.current_time_step()
  action_step = policy.action(time_step)
  next_time_step = environment.step(action_step.action)
  traj = trajectory.from_transition(time_step, action_step, next_time_step)

  # Add trajectory to the replay buffer
  buffer.add_batch(traj)

def collect_data(env, policy, buffer, steps):
  for _ in range(steps): 
    collect_step(env, policy, buffer) 

collect_data(train_env, random_policy, replay_buffer, initial_collect_steps)

dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,                                                                                                                                   
    sample_batch_size=batch_size, 
    num_steps=2).prefetch(3)"""

agent.train_step_counter.assign(0)

avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
returns = [avg_return]
                    
collect_op.run()#time_step=time_step)

print(" \n single trajectories:", iter(replay_buffer.as_dataset()).next())

dataset = replay_buffer.as_dataset(
  num_parallel_calls=3,                                                                                                                                   
  sample_batch_size=batch_size, 
  num_steps=2).prefetch(3)


iterator = iter(dataset)  
print("\n grouped trajectories:", iter(iterator.next()[0]))
                                                                                                   

for _ in range(num_iterations):
  # Collect a few steps using collect_policy and save to the replay buffer.
  #collect_data(train_env, agent.collect_policy, replay_buffer, collect_steps_per_iteration)
  collect_op.run(maximum_iterations=collect_steps_per_iteration)

  # Sample a batch of data from the buffer and update the agent's network.
  experience, unused_info = next(iterator)
  train_loss = agent.train(experience).loss

  step = agent.train_step_counter.numpy()
  print(step)

  """if step % log_interval == 0:
    print('step = {0}: loss = {1}'.format(step, train_loss))

  if step % eval_interval == 0:
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    print('step = {0}: Average Return = {1}'.format(step, avg_return))
    returns.append(avg_return)"""


def create_policy_eval_video(policy, filename, num_episodes=5, fps=30):
  filename = filename + ".mp4"
  with imageio.get_writer(filename, fps=fps) as video:
    for _ in range(num_episodes):
      time_step = eval_env.reset()
      video.append_data(eval_py_env.render())
      for _ in range(2000):
        action_step = policy.action(time_step)
        time_step = eval_env.step(action_step.action)
        video.append_data(eval_py_env.render())
  #return embed_mp4(filename)

create_policy_eval_video(agent.policy, "eval_video", 1)