import gym
import numpy as np

env = gym.make('FrozenLake-v0')
env.render()


def compute_value_function(policy, gamma=1.0):
    # initialize value table with zeros
    value_table = np.zeros(env.nS)
    # set the threshold
    threshold = 1e-10
    while True:
        # copy the value table to the updated_value_table
        updated_value_table = np.copy(value_table)
        # for each state in the environment, select the action according to the policy and compute the value table
        for state in range(env.nS):
            action = policy[state]
            # build the value table with the selected action
            value_table[state] = sum([trans_prob * (reward_prob + gamma * updated_value_table[next_state])
                                      for trans_prob, next_state, reward_prob, _ in env.P[state][action]])
        if (np.sum((np.fabs(updated_value_table - value_table))) <= threshold):
            break
    return value_table


def extract_policy(value_table, gamma=1.0):
    # Initialize the policy with zeros
    policy = np.zeros(env.observation_space.n)
    for state in range(env.observation_space.n):
        # initialize the Q table for a state
        Q_table = np.zeros(env.action_space.n)
        # compute Q value for all ations in the state
        for action in range(env.action_space.n):
            for next_sr in env.P[state][action]:
                trans_prob, next_state, reward_prob, _ = next_sr
                Q_table[action] += (trans_prob * (reward_prob + gamma * value_table[next_state]))
        # Select the action which has maximum Q value as an optimal action of the state
        policy[state] = np.argmax(Q_table)
    return policy


def policy_iteration(env, gamma=1.0):
    # Initialize policy with zeros
    old_policy = np.zeros(env.observation_space.n)
    no_of_iterations = 200000
    for i in range(no_of_iterations):
        # compute the value function
        new_value_function = compute_value_function(old_policy, gamma)
        # Extract new policy from the computed value function
        new_policy = extract_policy(new_value_function, gamma)
        # Then we check whether we have reached convergence i.e whether we found the optimal
        # policy by comparing old_policy and new policy if it same we will break the iteration
        # else we update old_policy with new_policy
        if (np.all(old_policy == new_policy)):
            print('Policy-Iteration converged at step %d.' % (i + 1))
            break
        old_policy = new_policy
    return new_policy

if __name__ == '__main__':
    print (policy_iteration(env))
