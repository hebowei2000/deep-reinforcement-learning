import tensorflow as tf
slim = tf.contrib.slim
import gin.tf


@gin.configurable('reward_net')
def reward_net(states, actions,
               for_critic_loss=False,
               num_reward_dims=1,
               states_hidden_layers=(400,),
               actions_hidden_layers=None,
               joint_hidden_layers=(300,),
               weight_decay=0.0001,
               normalizer_fn=None,
               activation_fn=tf.nn.relu,
               zero_obs=False,
               images=False):
  """Creates a reward net for the given states and actions.

  Args:
    states: (castable to tf.float32) a [batch_size, num_state_dims] tensor
      representing a batch of states.
    actions: (castable to tf.float32) a [batch_size, num_action_dims] tensor
      representing a batch of actions.
    num_reward_dims: Number of reward dimensions.
    states_hidden_layers: tuple of hidden layers units for states.
    actions_hidden_layers: tuple of hidden layers units for actions.
    joint_hidden_layers: tuple of hidden layers units after joining states
      and actions using tf.concat().
    weight_decay: Weight decay for l2 weights regularizer.
    normalizer_fn: Normalizer function, i.e. slim.layer_norm,
    activation_fn: Activation function, i.e. tf.nn.relu, slim.leaky_relu, ...
  Returns:
    A tf.float32 [batch_size] tensor of q values, or a tf.float32
      [batch_size, num_reward_dims] tensor of vector q values if
      num_reward_dims > 1.
  """
  with slim.arg_scope(
      [slim.fully_connected],
      activation_fn=activation_fn,
      normalizer_fn=normalizer_fn,
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=slim.variance_scaling_initializer(
          factor=1.0/3.0, mode='FAN_IN', uniform=True)):
    states=[states]
    actions=[actions]
    orig_states = tf.to_float(states)
    #states = tf.to_float(states)
    states = tf.concat([tf.to_float(states), tf.to_float(actions)], -1)  #TD3
    if images or zero_obs:
      states *= tf.constant([0.0] * 2 + [1.0] * (states.shape[1] - 2))  #LALA
    actions = tf.to_float(actions)
    if states_hidden_layers:
      states = slim.stack(states, slim.fully_connected, states_hidden_layers,
                          scope='states')
    if actions_hidden_layers:
      actions = slim.stack(actions, slim.fully_connected, actions_hidden_layers,
                           scope='actions')
    joint = tf.concat([states, actions], 1)
    if joint_hidden_layers:
      joint = slim.stack(joint, slim.fully_connected, joint_hidden_layers,
                         scope='joint')
    with slim.arg_scope([slim.fully_connected],
                        weights_regularizer=None,
                        weights_initializer=tf.random_uniform_initializer(
                            minval=-0.003, maxval=0.003)):
      value = slim.fully_connected(joint, num_reward_dims,
                                   activation_fn=None,
                                   normalizer_fn=None,
                                   scope='reward')
    if num_reward_dims == 1:
      value = tf.reshape(value, [-1])
    if not for_critic_loss and num_reward_dims > 1:
      value = tf.reduce_sum(
          value * tf.abs(orig_states[:, -num_reward_dims:]), -1)
  return value



@gin.configurable('completion_net')
def completion_net(states, actions,
               for_critic_loss=False,
               output_dim=2,
               states_hidden_layers=(400,),
               actions_hidden_layers=None,
               joint_hidden_layers=(300,),
               weight_decay=0.0001,
               normalizer_fn=None,
               activation_fn=tf.nn.relu,
               zero_obs=False,
               images=False):
  """Creates a completion net for the given states and actions.

  Args:
    states: (castable to tf.float32) a [batch_size, num_state_dims] tensor
      representing a batch of states.
    actions: (castable to tf.float32) a [batch_size, num_action_dims] tensor
      representing a batch of actions.
    num_reward_dims: Number of reward dimensions.
    states_hidden_layers: tuple of hidden layers units for states.
    actions_hidden_layers: tuple of hidden layers units for actions.
    joint_hidden_layers: tuple of hidden layers units after joining states
      and actions using tf.concat().
    weight_decay: Weight decay for l2 weights regularizer.
    normalizer_fn: Normalizer function, i.e. slim.layer_norm,
    activation_fn: Activation function, i.e. tf.nn.relu, slim.leaky_relu, ...
  Returns:
    A tf.float32 [batch_size] tensor of q values, or a tf.float32
      [batch_size, num_reward_dims] tensor of vector q values if
      num_reward_dims > 1.
  """
  #states=[states]
  #actions=[actions]
  with slim.arg_scope(
      [slim.fully_connected],
      activation_fn=activation_fn,
      normalizer_fn=normalizer_fn,
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=slim.variance_scaling_initializer(
          factor=1.0/3.0, mode='FAN_IN', uniform=True)):

    orig_states = tf.to_float(states)
    #states = tf.to_float(states)
    states = tf.concat([tf.to_float(states), tf.to_float(actions)], -1)  #TD3
    if images or zero_obs:
      states *= tf.constant([0.0] * 2 + [1.0] * (states.shape[1] - 2))  #LALA
    actions = tf.to_float(actions)
    if states_hidden_layers:
      states = slim.stack(states, slim.fully_connected, states_hidden_layers,
                          scope='states')
    if actions_hidden_layers:
      actions = slim.stack(actions, slim.fully_connected, actions_hidden_layers,
                           scope='actions')
    joint = tf.concat([states, actions], 1)
    if joint_hidden_layers:
      joint = slim.stack(joint, slim.fully_connected, joint_hidden_layers,
                         scope='joint')
    with slim.arg_scope([slim.fully_connected],
                        weights_regularizer=None,
                        weights_initializer=tf.random_uniform_initializer(
                            minval=-0.003, maxval=0.003)):
      value = slim.fully_connected(joint, output_dim,
                                   activation_fn=None,
                                   normalizer_fn=None,
                                   scope='completion')
  return value#tf.reshape(value, [1,2])


