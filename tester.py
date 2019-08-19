import tensorflow as tf
from environments import ant_maze_env
from environments import create_maze_env
from agents import circular_buffer
create_maze_env_fn = create_maze_env.create_maze_env
import agent
import math
from context import context
Buffer = circular_buffer.CircularBuffer
context_class = context.Context
agent_class = agent.UvfAgent
meta_agent_class = agent.MetaAgent

def main():
    test_compute_next_state()




def test_compute_next_state():
    sess = tf.Session()

    state = tf.random_uniform(shape=[15], minval=-5, maxval=5, dtype=tf.float32, name='state')
    state_numpy = state.eval(session=sess)
    #print(sess.run(state))
    state_numpy[0] = state_numpy[1] = 1
    state = tf.Variable(state_numpy)

    
    goal = tf.random_uniform(shape=[15], minval=-5, maxval=5, dtype=tf.float32, name='goal')
    goal_numpy = goal.eval(session=sess)
    #print(sess.run(state))
    goal_numpy[0] = 4
    goal_numpy[1] = 3
    goal = tf.Variable(goal_numpy)

    next_state = compute_next_state(state, goal, 0.8, 1/6)
    sess.run(tf.global_variables_initializer())
    result = sess.run(next_state)
    print(result)
    r = (result[1] - state_numpy[1]) ** 2 + (result[0] - state_numpy[0]) ** 2
    r **= 0.5
    print(r)
    [predicted_alpha, predicted_theta] = sess.run(inv_completion(state, goal, next_state))
    print(predicted_alpha, predicted_theta)


def compute_next_state(state, meta_action, alpha, theta):
    """
    Given state and meta-action, computes the predicted next_state
    using alpha and theta. 
    """
    r = alpha * tf.norm(meta_action[:2] - state[:2], ord=2)
    xy = [r * tf.math.cos(math.pi * theta), r * tf.math.sin(math.pi * theta)]
    next_state = state + tf.concat([xy, state[2:]], axis=0)
    return next_state 

def inv_completion(state, meta_action, next_state):
    """
    Given state and meta-action, computes alpha and theta using next_state.
    """
    ns_dist = tf.norm((next_state[:2] - state[:2]), ord=2)
    goal_dist = tf.norm((meta_action[:2] - state[:2]), ord=2)
    target_alpha = ns_dist / goal_dist
    target_theta = tf.math.acos((next_state[0] - state[0]) / ns_dist) / math.pi 
    return target_alpha, target_theta

def confidence(self, state, goal):
    buffer = Buffer(buffer_size=8)
    num_tensors = buffer.get_num_tensors()
    conf = 0
    if num_tensors :
      batch = buffer.get_random_batch(num_tensors)
      [states, meta_actions, next_states] = batch
      [predicted_alphas, predicted_thetas] = self.completion(states, meta_actions)
      [actual_alphas, actual_thetas] = self.inv_completion(states, meta_actions, next_states)
      for i in range(num_tensors):
        batch[i].append([predicted_alphas, predicted_thetas])
        batch[i].append([actual_alphas, actual_thetas])
        batch[i].append(tf.norm(states[i] - state, ord=2))
      sorted_batch = tf.sort(batch)  # sorts according to last axis
      weight_base = (num_tensors + num_tensors ** 2) / 2 
      for i in range(num_tensors):
        [predicted_alpha, predicted_theta] = sorted_batch[i][-3]
        [actual_alpha, actual_theta] = sorted_batch[i][-2]
        conf += (tf.abs(predicted_alpha - actual_alpha) + 
            tf.abs(predicted_theta - actual_theta)) / weight_base * i
    return conf

if __name__ == '__main__':
  main()