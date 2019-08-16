import tensorflow as tf
from environments import ant_maze_env
from environments import create_maze_env
create_maze_env_fn = create_maze_env.create_maze_env
import agent
from context import context
context_class = context.Context
agent_class = agent.UvfAgent
meta_agent_class = agent.MetaAgent
def main():
    environment = create_maze_env_fn(env_name='AntMaze')
    tf_env = create_maze_env.TFPyEnvironment(environment)
    with tf.variable_scope('meta_agent'):
      meta_agent = meta_agent_class(
        [tf_env.observation_spec()],
        [tf_env.action_spec()],
        tf_env,
      )
    with tf.variable_scope('uvf_agent'):
        uvf_agent = agent_class(
            [tf_env.observation_spec()],
            [tf_env.action_spec()],
            tf_env,
        )
    uvf_agent.set_meta_agent(agent=meta_agent)

    wrapped_environment = uvf_agent.get_env_base_wrapper(
              environment, mode='eval')
    while True:
        wrapped_environment._gym_env.render()
