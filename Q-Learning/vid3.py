'''
Building upon the vid1.py, this file contains
code for plotting graphs and analysing the policy
'''


import gym
import time
import numpy as np
import matplotlib.pyplot as plt

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
show_every = 3000   # render the environment after every ...
stats_every = 100
start_epsilon_decaying = 1
end_epsilon_decaying = episodes
epsilon_decay_value = epsilon / (end_epsilon_decaying - start_epsilon_decaying)

# A function to get the appropriate bin
def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_bin_size
    return tuple(discrete_state.astype(np.int))


discrete_os_size = [20] * len(env.observation_space.high) # No. of bins and this needn't be fixed
discrete_os_bin_size = (env.observation_space.high - env.observation_space.low) / discrete_os_size

size = discrete_os_size + [env.action_space.n]

q_table = np.random.uniform(low=-2, high=0, size=size)  # initializing the Q Table

ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

for episode in range(episodes):
    episode_reward = 0
    if episode % show_every == 0:
        print(episode)
        render = True
    else:
        render = False

    discrete_state = get_discrete_state(env.reset())  # Actually, this method call outputs the start state
    done = False
    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])  # greedy policy
        else:
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, done, _ = env.step(action)  # collecting info from the env
        episode_reward += reward
        new_discrete_state = get_discrete_state(new_state)
        if render:
            env.render()
            time.sleep(.01)     # This is to slow down the rendering
        if not done:
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

    ep_rewards.append(episode_reward)

    if not episode % stats_every:
        average_reward = sum(ep_rewards[-stats_every:])/stats_every
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-stats_every:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-stats_every:]))

        print(f"Episode: {episode} avg: {average_reward} max: {max(ep_rewards[-stats_every:])} min: {min(ep_rewards[-stats_every:])}")

env.close()

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label='avg')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label='max')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label='min')
plt.legend(loc=4)
plt.show()
