import random
import gym

"""
Say our agent is the driving the taxi. 
There are totally four locations and the agent has to pick up a passenger at one location and 
drop at the another. The agent will receive +20 points as a reward for successful drop off and 
-1 point for every time step it takes. The agent will also lose -10 points for illegal pickups and drops.
 So the goal of our agent is to learn to pick up and drop passengers at the correct location 
 in a short time without boarding any illegal passengers."""

env = gym.make('Taxi-v2')
env.render()

q = {}
for s in range(env.observation_space.n):
    for a in range(env.action_space.n):
        q[(s, a)] = 0.0


def update_q_table(prev_state, action, reward, nextstate, alpha, gamma):
    qa = max([q[(nextstate, a)] for a in range(env.action_space.n)])
    q[(prev_state, action)] += alpha * (reward + gamma * qa - q[(prev_state, action)])


def epsilon_greedy_policy(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return max(list(range(env.action_space.n)), key=lambda x: q[(state, x)])


if __name__ == '__main__':
    alpha = 0.4
    gamma = 0.999
    epsilon = 0.017
    for i in range(8000):
        r = 0
        prev_state = env.reset()
        while True:
            env.render()
            # In each state, we select the action by epsilon-greedy policy
            action = epsilon_greedy_policy(prev_state, epsilon)
            # then we perform the action and move to the next state, and receive the reward
            nextstate, reward, done, _ = env.step(action)
            # Next we update the Q value using our update_q_table function
            # which updates the Q value by Q learning update rule
            update_q_table(prev_state, action, reward, nextstate, alpha, gamma)
            # Finally we update the previous state as next state
            prev_state = nextstate
            # Store all the rewards obtained
            r += reward
            # we will break the loop, if we are at the terminal state of the episode
            if done:
                break
        print("total reward: ", r)
    env.close()
