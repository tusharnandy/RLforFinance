import gym
import time
import numpy as np

env = gym.make("MountainCar-v0")

'''
Each state_observation here is a list of two elements containing:
    1. position
    2. velocity

Since the observations are highly granular, we plan to discretize 'em by making bins
'''

alpha = 0.1         # learning rate
gamma = 0.95        # discount factor
episodes = 25000    # not rendering any of these
epsilon = 0.5       # parameter for behaviour policy
show_every = 2000   # render the environment after every ...

# In order to follow a formalism called GLIE:
# Greedy in the Limit with Infinite Exploration
# we need to decay the epsilon to a miniscule value (or 0), so that
# we finally end up with a greedy policy
start_epsilon_decaying = 1
end_epsilon_decaying = episodes // 2
epsilon_decay_value = epsilon / (end_epsilon_decaying - start_epsilon_decaying)

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_bin_size
    return tuple(discrete_state.astype(np.int))


discrete_os_size = [20] * len(env.observation_space.high) # No. of bins and this needn't be fixed
discrete_os_bin_size = (env.observation_space.high - env.observation_space.low) / discrete_os_size

size = discrete_os_size + [env.action_space.n]

q_table = np.random.uniform(low=-2, high=0, size=size)  # initializing the Q Table

for episode in range(episodes):
    if episode % show_every == 0:
        print(episode)
        render = True
    else:
        render = False

    discrete_state = get_discrete_state(env.reset())  # Actually, this method call outputs the start state
    done = False
    while not done:
        # choosing action according to an eps-greedy policy
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])  # greedy policy
        else:
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, done, _ = env.step(action)  # collecting info from the env
        new_discrete_state = get_discrete_state(new_state)

        if render:
            env.render()
            time.sleep(.01)     # This is to slow down the rendering
        if not done:
            # Updating the LUT according to SARSAMAX and TD(1)
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]
            new_q = (1 - alpha)*current_q + alpha*(reward + gamma*max_future_q)
            q_table[discrete_state + (action,)] = new_q
        elif new_state[0] >= env.goal_position:
            print(f"We made it on episode: {episode}")
            q_table[discrete_state + (action,)] = 0

        discrete_state = new_discrete_state
    if end_epsilon_decaying >= episode >= start_epsilon_decaying:
        epsilon -= epsilon_decay_value

env.close()
