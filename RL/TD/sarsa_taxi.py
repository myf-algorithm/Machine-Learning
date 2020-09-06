import random
import gym

env = gym.make('Taxi-v2')
env.render()
Q = {}

for s in range(env.observation_space.n):
    for a in range(env.action_space.n):
        Q[(s, a)] = 0.0


def epsilon_greedy(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return max(list(range(env.action_space.n)), key=lambda x: Q[(state, x)])


if __name__ == '__main__':
    alpha = 0.85  # TD learning rate
    gamma = 0.90  # discount factor
    epsilon = 0.8  # epsilon value in epsilon greedy policy
    for i in range(4000):
        # we store cumulative reward of each episodes in r
        r = 0
        # initialize the state,
        state = env.reset()
        # select the action using epsilon-greedy policy
        action = epsilon_greedy(state, epsilon)
        while True:
            # env.render()
            # then we perform the action and move to the next state, and receive the reward
            nextstate, reward, done, _ = env.step(action)
            # again, we select the next action using epsilon greedy policy
            nextaction = epsilon_greedy(nextstate, epsilon)
            # we calculate the Q value of previous state using our update rule
            Q[(state, action)] += alpha * (reward + gamma * Q[(nextstate, nextaction)] - Q[(state, action)])
            # finally we update our state and action with next action and next state
            action = nextaction
            state = nextstate
            # store the rewards
            r += reward
            # we will break the loop, if we are at the terminal state of the episode
            if done:
                break
        print("total reward: ", r)
    env.close()
