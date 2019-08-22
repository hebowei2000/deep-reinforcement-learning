# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Script for training an RL agent using the UVF algorithm.

To run locally: See run_train.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import tensorflow as tf
slim = tf.contrib.slim

import gin.tf
# pylint: disable=unused-import
import train_utils
import agent as agent_
from agents import circular_buffer
from utils import utils as uvf_utils
from environments import create_maze_env
# pylint: enable=unused-import

flags = tf.app.flags

FLAGS = flags.FLAGS
flags.DEFINE_string('goal_sample_strategy', 'sample',
                    'None, sample, FuN')

LOAD_PATH = None


def collect_experience(tf_env, agent, meta_agent, state_preprocess, 
                       replay_buffer, meta_replay_buffer,
                       action_fn, meta_action_fn, num_leaps,
                       environment_steps, num_episodes, num_resets,
                       episode_rewards, episode_meta_rewards,
                       store_context, max_steps_per_episode,
                       disable_agent_reset, c_min, 
                       reward_optimizer, completion_optimizer,
                       global_step, update_ops, summarize_gradients,
                        clip_gradient_norm,):
  """Collect experience in a tf_env into a replay_buffer using action_fn.

  Args:
    tf_env: A TFEnvironment.
    agent: A UVF agent.
    meta_agent: A Meta Agent.
    replay_buffer: A Replay buffer to collect experience in.
    meta_replay_buffer: A Replay buffer to collect meta agent experience in.
    action_fn: A function to produce actions given current state.
    meta_action_fn: A function to produce meta actions given current state.
    environment_steps: A variable to count the number of steps in the tf_env.
    num_episodes: A variable to count the number of episodes.
    num_resets: A variable to count the number of resets.
    store_context: A boolean to check if store context in replay.
    disable_agent_reset: A boolean that disables agent from resetting.

  Returns:
    A collect_experience_op that excute an action and store into the
    replay_buffers
  """
  def increment_step():
    return environment_steps.assign_add(1)

  def increment_many_steps():
    return environment_steps.assign_add(agent.tf_context.meta_action_every_n)

  def increment_episode():
    return num_episodes.assign_add(1)

  def increment_leaps():
    return num_leaps.assign_add(1) 

  def increment_reset():
    return num_resets.assign_add(1)

  def update_episode_rewards(context_reward, meta_reward, reset): #
    new_episode_rewards = tf.concat(
        [episode_rewards[:1] + context_reward, episode_rewards[1:]], 0) 
    new_episode_meta_rewards = tf.concat(
        [episode_meta_rewards[:1] + meta_reward,
         episode_meta_rewards[1:]], 0)
    return tf.group(
        episode_rewards.assign(
            tf.cond(reset,
                    lambda: tf.concat([[0.], episode_rewards[:-1]], 0),
                    lambda: new_episode_rewards)),
        episode_meta_rewards.assign(
            tf.cond(reset,
                    lambda: tf.concat([[0.], episode_meta_rewards[:-1]], 0),
                    lambda: new_episode_meta_rewards)))

  def no_op_int():
    return tf.constant(0, dtype=tf.int64)


  tf_env.start_collect() # no effect
  state = tf_env.current_obs() # s
  state_repr = state_preprocess(state) # f(s)
  
  """ start temporal leap """
  meta_action_every_n = agent.tf_context.meta_action_every_n
  
  if store_context:
    context = [tf.identity(var) + tf.zeros_like(var) for var in agent.context_vars]
    meta_context = [tf.identity(var) + tf.zeros_like(var) for var in meta_agent.context_vars]
  else:
    # never executes
    context = []
    meta_context = []

  meta_action = tf.to_float(tf.concat(context, -1))
  #meta_action = context
  
  with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
      starting_state = tf.get_variable('state_var', state.shape, state.dtype)
      meta_action_var = tf.get_variable('meta_action_var',
                                          meta_action.shape, meta_action.dtype)

  confidence = meta_agent.confidence(state, context[0])
  
  is_new_goal = environment_steps % meta_action_every_n == 0
  
  
  
  def leap():

    ep_upd = agent._every_n_steps(None, None, None, None, environment_steps, None) 
    #meta_action_var = meta_action
    # predict completion and compute next_state
    
    next_state = meta_agent.compute_next_state(state, meta_action_var, alpha, theta)
    
    increment_step_op = tf.cond(ep_upd, increment_many_steps, no_op_int)
    increment_ep_op = tf.cond(ep_upd, increment_episode, no_op_int)
    increment_op = [increment_step_op, increment_ep_op]
    
    # experience for meta_buffer
    meta_transition = [state, meta_action_var] 
    buffer_push_op = meta_replay_buffer.add(meta_transition)
    #pos = tf_env._env._gym_env.wrapped_env.get_xy()
    pos = next_state[:2]
    tf_env._env._gym_env.wrapped_env.set_xy(pos)
    

    with tf.control_dependencies([increment_op]):
      # increment time step 
      tf_env._env._gym_env.t += meta_action_every_n
      agent.tf_context.t += meta_action_every_n
    
    def add_gaussian_noise(data, variance):
      stddev = tf.sqrt(variance)
      noise = tf.random_normal(shape=tf.shape(data), mean=0.0, stddev=stddev, dtype=tf.float32) 
      return data + noise

    with tf.control_dependencies(buffer_push_op):
      comp_train_op = None
      if meta_agent.mini_buffer.buffer_size == meta_agent.mini_buffer.get_num_tensors:
        # use popped to optimize completion net
        popped = tf.identity(meta_agent.mini_buffer._tensors[meta_agent.mini_buffer.get_position()])
        target = popped[-1]  # [alpha, theta]
        
        comp_loss = meta_agent.completion_loss(state, meta_action_var, target)
       
        comp_train_op = slim.learning.create_train_op(
          comp_loss,
          completion_optimizer, #
          global_step=global_step,
          update_ops=None,
          summarize_gradients=summarize_gradients,
          clip_gradient_norm=clip_gradient_norm,
          variables_to_train=meta_agent.get_trainable_completion_vars(),)

      # experience for mini_buffer
      meta_transition.append(add_gaussian_noise([alpha, theta], confidence))
      buffer_push_op = meta_agent.mini_buffer.add(meta_transition) 

    with tf.control_dependencies(buffer_push_op):
      # manually set new goal for low level agent 
      # TODO: experiment: does the agent take care of itself? 
      low_level_context = meta_action_fn(next_state, context=None)
      agent_context_upd = [tf.assign(var, value)
                 for var, value in zip(agent.context_vars, low_level_context)]

    state_var_upd = tf.assign(starting_state, next_state)
    meta_action_var_upd = tf.assign(meta_action_var, low_level_context)
    
    return tf.group(comp_train_op, state_var_upd, meta_action_var_upd *agent_context_upd)
  
  
  def walk():
    action = action_fn(state, context=None) # a

    starting_state_repr = state_preprocess(starting_state)

    with tf.control_dependencies([state]):
      transition_type, reward, discount = tf_env.step(action)  # takes step and enters new state

    # tf.bool that tells if the transition_type is TRANSITION or FINAL_TRANSITION.
    step_cond = agent.step_cond_fn(state, action,       
                                   transition_type,
                                   environment_steps, num_episodes)

    # tf.bool that tells if the transition_type is RESTARTING.
    reset_episode_cond = agent.reset_episode_cond_fn(   
        state, action,
        transition_type, environment_steps, num_episodes)

    # tf.bool that is always false
    reset_env_cond = agent.reset_env_cond_fn(state, action,
                                             transition_type,
                                             environment_steps, num_episodes)

    # if transition_type is TRANSITION or FINAL_TRANSITION, add one to environment_steps, otherwise do nothing.                               
    increment_step_op = tf.cond(step_cond, increment_step, no_op_int)

    # if transition_type is RESTARTING, add one to num_episodes, otherwise do nothing.                              
    increment_episode_op = tf.cond(reset_episode_cond, increment_episode,
                                   no_op_int)

    # never executes.
    increment_reset_op = tf.cond(reset_env_cond, increment_reset, no_op_int)

    # a group of tf.int64's
    increment_op = tf.group(increment_step_op, increment_episode_op,
                            increment_reset_op)

    # observe next state & repr
    with tf.control_dependencies([increment_op, reward, discount]):
      next_state = tf_env.current_obs()
      next_state_repr = state_preprocess(next_state)
      next_reset_episode_cond = tf.logical_or(
          agent.reset_episode_cond_fn(
              state, action,
              transition_type, environment_steps, num_episodes),
          tf.equal(discount, 0.0))

    ls = [next_state] + context + meta_context

    with tf.control_dependencies(ls):
      if disable_agent_reset:
        assert False # never executes
        collect_experience_ops = [tf.no_op()]  # don't reset agent
      else:
        collect_experience_ops = agent.cond_begin_episode_op(
            tf.logical_not(reset_episode_cond),
            [state, starting_state, action, reward, next_state,
             state_repr, starting_state_repr, next_state_repr],
            mode='explore', meta_action_fn=meta_action_fn)
        context_reward, meta_reward = collect_experience_ops
        collect_experience_ops = list(collect_experience_ops) # convert tuple to list
        collect_experience_ops.append(
            update_episode_rewards(tf.reduce_sum(context_reward), meta_reward,
                                   reset_episode_cond))

    with tf.control_dependencies(collect_experience_ops):
      transition = [state, starting_state, action, reward, discount, next_state] 

      meta_action = tf.to_float(
          tf.concat(context, -1))  # Meta agent action is low-level context

      meta_end = tf.logical_and(  # End of meta-transition.
          tf.equal(agent.tf_context.t % meta_action_every_n, 1),
          agent.tf_context.t > 1)

      with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        state_var = tf.get_variable('state_var', state.shape, state.dtype)
        reward_var = tf.get_variable('reward_var', reward.shape, reward.dtype)
        meta_action_var = tf.get_variable('meta_action_var',
                                          meta_action.shape, meta_action.dtype)
        meta_context_var = [
            tf.get_variable('meta_context_var%d' % idx,
                            meta_context[idx].shape, meta_context[idx].dtype)
            for idx in range(len(meta_context))]


      meta_reward = tf.identity(meta_reward) + tf.zeros_like(meta_reward)
      meta_reward = tf.reshape(meta_reward, reward.shape)

      # scale down reward
      reward = 0.1 * meta_reward
      meta_transition = [state_var, meta_action_var]
      mini_transition = [state_var, meta_action_var, next_state]
      #meta_transition.extend([states_var, actions])
      if store_context:  # store current and next context into replay
        transition += context + list(agent.context_vars)
        meta_transition += meta_context_var + list(meta_agent.context_vars)

      meta_step_cond = tf.squeeze(tf.logical_and(step_cond, tf.logical_or(next_reset_episode_cond, meta_end)))
      collect_experience_op = tf.group(
          replay_buffer.maybe_add(transition, step_cond),
          meta_replay_buffer.maybe_add(meta_transition, meta_step_cond),
          #meta_agent.mini_buffer.maybe_add(mini_transition, meta_step_cond),
      )

    with tf.control_dependencies([collect_experience_op]):
      collect_experience_op = tf.cond(reset_env_cond,    # alwasy evals to false
                                      tf_env.reset,
                                      tf_env.current_time_step)
      meta_period = tf.equal(agent.tf_context.t % meta_action_every_n, 1)                                                               
    
    
    
    def train_comp():
      comp_train_op = tf.constant(0.0)
      
      if meta_agent.mini_buffer.buffer_size == meta_agent.mini_buffer.get_num_tensors:
        # use popped to optimize completion net
        popped = tf.identity(meta_agent.mini_buffer._tensors[meta_agent.mini_buffer.get_position()])
        target = popped[-1]  # [alpha, theta]

        comp_loss = meta_agent.completion_loss(state, meta_action, target)

        comp_train_op = slim.learning.create_train_op(
          comp_loss,
          completion_optimizer, 
          global_step=global_step,
          update_ops=None,
          summarize_gradients=summarize_gradients,
          clip_gradient_norm=clip_gradient_norm,
          variables_to_train=meta_agent.get_trainable_completion_vars(),)
      
      # experience for mini_buffer
      buffer_push_op = meta_agent.mini_buffer.add(mini_transition)

      with tf.control_dependencies([buffer_push_op]):
        return comp_train_op
    
    def train_reward():
      meta_reward_loss = meta_agent.reward_loss(state_var, meta_action_var, reward_var + reward, next_state),
      tabvars = meta_agent.get_trainable_reward_vars()
      reward_train_op = slim.learning.create_train_op(
          meta_reward_loss[0],
          reward_optimizer,
          global_step=global_step,
          update_ops=update_ops,
          summarize_gradients=summarize_gradients,
          clip_gradient_norm=clip_gradient_norm,
          variables_to_train=tabvars,)
      return reward_train_op

    comp_train_op = tf.cond(meta_period, train_comp, lambda: tf.constant(0.0))

       
    reward_train_op = tf.cond(meta_period, train_reward, lambda: tf.constant(0.0))
    
    state_var_upd = tf.assign(
        state_var,
        tf.cond(meta_period, lambda: next_state, lambda: state_var))
    reward_var_upd = tf.assign(
        reward_var,
        tf.cond(meta_period,
                lambda: tf.zeros_like(reward_var),
                lambda: reward_var + reward))
    meta_action = tf.to_float(tf.concat(agent.context_vars, -1))
    meta_action_var_upd = tf.assign(
        meta_action_var,
        tf.cond(meta_period, lambda: meta_action, lambda: meta_action_var))
    meta_context_var_upd = [
        tf.assign(
            meta_context_var[idx],
            tf.cond(meta_period,
                    lambda: meta_agent.context_vars[idx],
                    lambda: meta_context_var[idx]))
        for idx in range(len(meta_context))]

    return tf.group(
      collect_experience_op,
      state_var_upd,
      reward_var_upd,
      meta_action_var_upd,
      reward_train_op,
      comp_train_op,
      *meta_context_var_upd)


  return walk()
  #return tf.cond(tf.math.logical_and(confidence > c_min, is_new_goal), leap, walk) 
  #TODO: change to not only walk


def sample_best_meta_actions(state_reprs, next_state_reprs, prev_meta_actions,
                             low_states, low_actions, low_state_reprs,
                             inverse_dynamics, uvf_agent, k=10):
  """Return meta-actions which approximately maximize low-level log-probs."""
  sampled_actions = inverse_dynamics.sample(state_reprs, next_state_reprs, k, prev_meta_actions)
  sampled_actions = tf.stop_gradient(sampled_actions)
  sampled_log_probs = tf.reshape(uvf_agent.log_probs(
      tf.tile(low_states, [k, 1, 1]),
      tf.tile(low_actions, [k, 1, 1]),
      tf.tile(low_state_reprs, [k, 1, 1]),
      [tf.reshape(sampled_actions, [-1, sampled_actions.shape[-1]])]),
                                 [k, low_states.shape[0],
                                  low_states.shape[1], -1])
  fitness = tf.reduce_sum(sampled_log_probs, [2, 3])
  best_actions = tf.argmax(fitness, 0)
  actions = tf.gather_nd(
      sampled_actions,
      tf.stack([best_actions,
                tf.range(prev_meta_actions.shape[0], dtype=tf.int64)], -1))
  return actions


@gin.configurable
def train_uvf(train_dir,
              environment=None,
              num_bin_actions=3,
              agent_class=None,
              meta_agent_class=None,
              state_preprocess_class=None,
              inverse_dynamics_class=None,
              exp_action_wrapper=None,
              replay_buffer=None,
              meta_replay_buffer=None,
              mini_buffer=None,
              replay_num_steps=1,
              meta_replay_num_steps=1,
              critic_optimizer=None,
              actor_optimizer=None,
              meta_critic_optimizer=None,
              meta_actor_optimizer=None,
              reward_optimizer=None,
              completion_optimizer=None,
              repr_optimizer=None,
              relabel_contexts=False,
              meta_relabel_contexts=False,
              batch_size=64,
              repeat_size=0,
              num_episodes_train=2000,
              initial_episodes=2,
              initial_steps=None,
              num_updates_per_observation=1,
              num_collect_per_update=1,
              num_collect_per_meta_update=1,
              gamma=1.0,
              meta_gamma=1.0,
              reward_scale_factor=1.0,
              target_update_period=1,
              should_stop_early=None,
              clip_gradient_norm=0.0,
              summarize_gradients=False,
              debug_summaries=False,
              log_every_n_steps=100,
              prefetch_queue_capacity=2,
              policy_save_dir='policy',
              save_policy_every_n_steps=1000,
              save_policy_interval_secs=0,
              replay_context_ratio=0.0,
              next_state_as_context_ratio=0.0,
              state_index=0,
              zero_timer_ratio=0.0,
              timer_index=-1,
              debug=False,
              max_policies_to_save=None,
              max_steps_per_episode=None,
              load_path=LOAD_PATH):
  """Train an agent."""
  tf_env = create_maze_env.TFPyEnvironment(environment)
  observation_spec = [tf_env.observation_spec()]
  action_spec = [tf_env.action_spec()] 

  max_steps_per_episode = max_steps_per_episode or tf_env.pyenv.max_episode_steps # environment specification

  assert max_steps_per_episode, 'max_steps_per_episode need to be set'

  if initial_steps is None:
    initial_steps = initial_episodes * max_steps_per_episode


  assert agent_class.ACTION_TYPE == 'continuous'
  
  #   agent_class.ACTION_TYPE == 'discrete':
  #  assert False # quit
  #else:
   
  # must be both continuous
  assert agent_class.ACTION_TYPE == meta_agent_class.ACTION_TYPE 

  with tf.variable_scope('meta_agent'):
    meta_agent = meta_agent_class(
        observation_spec,
        action_spec,
        tf_env,
        debug_summaries=debug_summaries)
    meta_agent.set_replay(replay=meta_replay_buffer)
    meta_agent.set_mini_buffer(mini_buffer)

  with tf.variable_scope('uvf_agent'):
    uvf_agent = agent_class(
        observation_spec,
        action_spec,
        tf_env,
        debug_summaries=debug_summaries)
    uvf_agent.set_meta_agent(agent=meta_agent)
    uvf_agent.set_replay(replay=replay_buffer)

  with tf.variable_scope('state_preprocess'):
    state_preprocess = state_preprocess_class()

  with tf.variable_scope('inverse_dynamics'):
    inverse_dynamics = inverse_dynamics_class(
        meta_agent.sub_context_as_action_specs[0])

  # Create counter variables
  global_step = tf.contrib.framework.get_or_create_global_step()
  num_episodes = tf.Variable(0, dtype=tf.int64, name='num_episodes')
  num_resets = tf.Variable(0, dtype=tf.int64, name='num_resets')
  num_updates = tf.Variable(0, dtype=tf.int64, name='num_updates')
  num_leaps = tf.Variable(0, dtype=tf.int64, name='num_leaps')
  num_meta_updates = tf.Variable(0, dtype=tf.int64, name='num_meta_updates')
  episode_rewards = tf.Variable([0.] * 100, name='episode_rewards')
  episode_meta_rewards = tf.Variable([0.] * 100, name='episode_meta_rewards')

  # Create counter variables summaries
  train_utils.create_counter_summaries([
      ('environment_steps', global_step),
      ('num_episodes', num_episodes),
      ('num_resets', num_resets),
      ('num_updates', num_updates),
      ('num_leaps', num_leaps),
      ('num_meta_updates', num_meta_updates),
      ('replay_buffer_adds', replay_buffer.get_num_adds()),
      ('meta_replay_buffer_adds', meta_replay_buffer.get_num_adds()),
  ])

  tf.summary.scalar('avg_episode_rewards',
                    tf.reduce_mean(episode_rewards[1:]))
  tf.summary.scalar('avg_episode_meta_rewards',
                    tf.reduce_mean(episode_meta_rewards[1:]))
  tf.summary.scalar('num_leaps',num_leaps)
  tf.summary.histogram('episode_rewards', episode_rewards[1:])
  tf.summary.histogram('episode_meta_rewards', episode_meta_rewards[1:])

  # Create init ops
  action_fn = uvf_agent.action # actor
  action_fn = uvf_agent.add_noise_fn(action_fn, global_step=None) 
  meta_action_fn = meta_agent.action
  meta_action_fn = meta_agent.add_noise_fn(meta_action_fn, global_step=None)
  meta_actions_fn = meta_agent.actions # returns actor net
  meta_actions_fn = meta_agent.add_noise_fn(meta_actions_fn, global_step=None)
  
  init_collect_experience_op = collect_experience(
      tf_env,
      uvf_agent,
      meta_agent,
      state_preprocess,
      replay_buffer,
      meta_replay_buffer,
      action_fn,
      meta_action_fn,
      environment_steps=global_step,
      num_leaps=num_leaps,
      num_episodes=num_episodes,
      num_resets=num_resets,
      episode_rewards=episode_rewards,
      episode_meta_rewards=episode_meta_rewards,
      store_context=True,
      disable_agent_reset=False,
      reward_optimizer=reward_optimizer,
      completion_optimizer=completion_optimizer,
      max_steps_per_episode = max_steps_per_episode,
      global_step = num_meta_updates, 
      update_ops = None, 
      summarize_gradients = summarize_gradients,
      clip_gradient_norm = clip_gradient_norm, 
      c_min = 0.75

  )

  # Create train ops
  collect_experience_op = collect_experience(
      tf_env,
      uvf_agent,
      meta_agent,
      state_preprocess,
      replay_buffer,
      meta_replay_buffer,
      action_fn,
      meta_action_fn,
      environment_steps=global_step,
      num_leaps=num_leaps,
      num_episodes=num_episodes,
      num_resets=num_resets,
      episode_rewards=episode_rewards,
      episode_meta_rewards=episode_meta_rewards,
      store_context=True,
      disable_agent_reset=False,
      reward_optimizer=reward_optimizer,
      completion_optimizer=completion_optimizer,
      max_steps_per_episode = max_steps_per_episode,
      global_step = num_meta_updates, 
      update_ops = None, 
      summarize_gradients = summarize_gradients,
      clip_gradient_norm = clip_gradient_norm, 
      c_min = 0.75
  )

  train_op_list = []
  repr_train_op = tf.constant(0.0)
  for mode in ['meta', 'nometa']:
    if mode == 'meta':
      agent = meta_agent
      buff = meta_replay_buffer
      critic_opt = meta_critic_optimizer
      actor_opt = meta_actor_optimizer
      relabel = meta_relabel_contexts
      num_steps = meta_replay_num_steps
      my_gamma = meta_gamma,
      n_updates = num_meta_updates
    else:
      agent = uvf_agent
      buff = replay_buffer
      critic_opt = critic_optimizer
      actor_opt = actor_optimizer
      relabel = relabel_contexts
      num_steps = replay_num_steps
      my_gamma = gamma
      n_updates = num_updates

    with tf.name_scope(mode):
      batch = buff.get_random_batch(batch_size, num_steps=num_steps)
      if mode == 'nometa': 
        states, actions, rewards, next_states = batch[:4]      
      else:
        states, actions = batch[:2]
        # relabel next_state
        # relabel reward 
        ret = meta_agent.completion(states, actions)
        alphas = ret[0]
        thetas = ret[1]
        next_states = meta_agent.compute_next_state(states, actions, alphas, thetas)
        rewards = meta_agent.reward_net(states)       
      with tf.name_scope('Reward'):
        tf.summary.scalar('average_step_reward', tf.reduce_mean(rewards))
      rewards *= reward_scale_factor
      batch_queue = slim.prefetch_queue.prefetch_queue(
          [states, actions, rewards, next_states] + batch[4:],
          capacity=prefetch_queue_capacity,
          name='batch_queue')  

      batch_dequeue = batch_queue.dequeue()
      if repeat_size > 0:    # repeat_size is always 0
        batch_dequeue = [
            tf.tile(batch, (repeat_size+1,) + (1,) * (batch.shape.ndims - 1))
            for batch in batch_dequeue
        ]
        batch_size *= (repeat_size + 1)
      states, actions, rewards, next_states = batch_dequeue[:4]
      if mode == 'meta':
        low_states = batch_dequeue[6]
        low_actions = batch_dequeue[7]
        low_state_reprs = state_preprocess(low_states)
      state_reprs = state_preprocess(states)
      next_state_reprs = state_preprocess(next_states)
      #starting_states_reprs = state_preprocess(starting_states) TODO: starting_state

      # DELETED: meta-action relabel

      if state_preprocess.trainable and mode == 'meta':
        # Representation learning is based on meta-transitions, but is trained
        # along with low-level policy updates.
        repr_loss, _, _ = state_preprocess.loss(states, next_states, low_actions, low_states)
        repr_train_op = slim.learning.create_train_op(
            repr_loss,
            repr_optimizer,
            global_step=None,
            update_ops=None,
            summarize_gradients=summarize_gradients,
            clip_gradient_norm=clip_gradient_norm,
            variables_to_train=state_preprocess.get_trainable_vars(),)

      # Get contexts for training
      contexts, next_contexts = agent.sample_contexts(
          mode='train', batch_size=batch_size,
          state=states, next_state=next_states,
      )

      # if nometa, contexts is replaced with contexts
      #  acquired from transition experience. 
      if not relabel:  # Re-label context (in the style of TDM or HER).
        contexts, next_contexts = (
            batch_dequeue[-2*len(contexts):-1*len(contexts)],
            batch_dequeue[-1*len(contexts):]) 


      merged_states = agent.merged_states(states, contexts)
      merged_next_states = agent.merged_states(next_states, next_contexts)
      

      # When computing reward for meta, use state_dim=30
      # When computing reward for nometa, use repr_state_dim=15
      if mode == 'nometa':
        context_rewards, context_discounts = agent.compute_rewards(
            'train', state_reprs, None, actions, rewards,next_state_reprs, contexts)
      elif mode == 'meta': # Meta-agent uses sum of rewards, not context-specific rewards.
        _, context_discounts = agent.compute_rewards(
            'train', states, None, actions, rewards, next_states, contexts)
        context_rewards = rewards # does not use returned rewards
      
      if agent.gamma_index is not None:
        context_discounts *= tf.cast(
            tf.reshape(contexts[agent.gamma_index], (-1,)),
            dtype=context_discounts.dtype)
      else: context_discounts *= my_gamma

      #critic_loss = agent.critic_loss(merged_states, actions,
      #                                context_rewards, context_discounts,
      #                                merged_next_states)
      critic_loss = agent.critic_loss(merged_states, actions,
                                      context_rewards, context_discounts,
                                      merged_next_states)
      critic_loss = tf.reduce_mean(critic_loss)

      actor_loss = agent.actor_loss(merged_states, actions,
                                    context_rewards, context_discounts,
                                    merged_next_states)
      actor_loss *= tf.to_float(  # Only update actor every N steps.
          tf.equal(n_updates % target_update_period, 0))

      critic_train_op = slim.learning.create_train_op(
          critic_loss,
          critic_opt,
          global_step=n_updates,
          update_ops=None,
          summarize_gradients=summarize_gradients,
          clip_gradient_norm=clip_gradient_norm,
          variables_to_train=agent.get_trainable_critic_vars(),)
      critic_train_op = uvf_utils.tf_print(
          critic_train_op, [critic_train_op],
          message='critic_loss',
          print_freq=1000,
          name='critic_loss')
      train_op_list.append(critic_train_op)
      if actor_loss is not None:
        actor_train_op = slim.learning.create_train_op(
            actor_loss,
            actor_opt,
            global_step=None,
            update_ops=None,
            summarize_gradients=summarize_gradients,
            clip_gradient_norm=clip_gradient_norm,
            variables_to_train=agent.get_trainable_actor_vars(),)
        actor_train_op = uvf_utils.tf_print(
            actor_train_op, [actor_train_op],
            message='actor_loss',
            print_freq=1000,
            name='actor_loss')
        train_op_list.append(actor_train_op)

  assert len(train_op_list) == 4
  # Update targets should happen after the networks have been updated.
  with tf.control_dependencies(train_op_list[2:]):
    update_targets_op = uvf_utils.periodically(
        uvf_agent.update_targets, target_update_period, 'update_targets')
  if meta_agent is not None:
    with tf.control_dependencies(train_op_list[:2]):
      update_meta_targets_op = uvf_utils.periodically(
          meta_agent.update_targets, target_update_period, 'update_targets')

  assert_op = tf.Assert(  # Hack to get training to stop.
      tf.less_equal(global_step, 200 + num_episodes_train * max_steps_per_episode),
      [global_step])
  with tf.control_dependencies([update_targets_op, assert_op]):
    train_op = tf.add_n(train_op_list[2:], name='post_update_targets')
    # Representation training steps on every low-level policy training step.
    train_op += repr_train_op
  with tf.control_dependencies([update_meta_targets_op, assert_op]):
    meta_train_op = tf.add_n(train_op_list[:2],
                             name='post_update_meta_targets')

  if debug_summaries:
    train_.gen_debug_batch_summaries(batch)
    slim.summaries.add_histogram_summaries(
        uvf_agent.get_trainable_critic_vars(), 'critic_vars')
    slim.summaries.add_histogram_summaries(
        uvf_agent.get_trainable_actor_vars(), 'actor_vars')

  train_ops = train_utils.TrainOps(train_op, meta_train_op,
                                   collect_experience_op)

  policy_save_path = os.path.join(train_dir, policy_save_dir, 'model.ckpt')
  policy_vars = uvf_agent.get_actor_vars() + meta_agent.get_actor_vars() + [
      global_step, num_episodes, num_resets
  ] + list(uvf_agent.context_vars) + list(meta_agent.context_vars) + state_preprocess.get_trainable_vars()
  # add critic vars, since some test evaluation depends on them
  policy_vars += uvf_agent.get_trainable_critic_vars() + meta_agent.get_trainable_critic_vars()
  policy_saver = tf.train.Saver(
      policy_vars, max_to_keep=max_policies_to_save, sharded=False)

  lowlevel_vars = (uvf_agent.get_actor_vars() +
                   uvf_agent.get_trainable_critic_vars() +
                   state_preprocess.get_trainable_vars())
  lowlevel_saver = tf.train.Saver(lowlevel_vars)

  def policy_save_fn(sess):
    policy_saver.save(
        sess, policy_save_path, global_step=global_step, write_meta_graph=False)
    if save_policy_interval_secs > 0:
      tf.logging.info(
          'Wait %d secs after save policy.' % save_policy_interval_secs)
      time.sleep(save_policy_interval_secs)

  train_step_fn = train_utils.TrainStep(
      max_number_of_steps=num_episodes_train * max_steps_per_episode + 100,
      num_updates_per_observation=num_updates_per_observation,
      num_collect_per_update=num_collect_per_update,
      num_collect_per_meta_update=num_collect_per_meta_update,
      log_every_n_steps=log_every_n_steps,
      policy_save_fn=policy_save_fn,
      save_policy_every_n_steps=save_policy_every_n_steps,
      should_stop_early=should_stop_early).train_step

  local_init_op = tf.local_variables_initializer()
  init_targets_op = tf.group(uvf_agent.update_targets(1.0),
                             meta_agent.update_targets(1.0))

  def initialize_training_fn(sess):
    """Initialize training function."""
    sess.run(local_init_op)
    sess.run(init_targets_op)
    if load_path:
      tf.logging.info('Restoring low-level from %s' % load_path)
      lowlevel_saver.restore(sess, load_path)
    global_step_value = sess.run(global_step)
    assert global_step_value == 0, 'Global step should be zero.'
    collect_experience_call = sess.make_callable(
        init_collect_experience_op)

    for _ in range(initial_steps):
      collect_experience_call()

  train_saver = tf.train.Saver(max_to_keep=2, sharded=True)
  tf.logging.info('train dir: %s', train_dir)
  #slim.learning.train is the main training process of the networks
  return slim.learning.train(
      train_ops,
      train_dir,
      train_step_fn=train_step_fn,
      save_interval_secs=FLAGS.save_interval_secs,
      saver=train_saver,
      log_every_n_steps=0,
      global_step=global_step,
      master="",
      is_chief=(FLAGS.task == 0),
      save_summaries_secs=FLAGS.save_summaries_secs,
      init_fn=initialize_training_fn)
