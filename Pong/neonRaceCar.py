"""
A bare bones examples of optimizing a black-box function (f) using
Natural Evolution Strategies (NES), where the parameter distribution is a
gaussian of fixed standard deviation.
"""

import random
import gym
import universe
import go_vncdriver
import numpy as np

env = gym.make('wob.mini.CircleCenter-v0')
env.configure(remotes=1)  # create one flashgames Docker container
observation_n = env.reset()

np.random.seed(0)
t = 0
# the function we want to optimize
u = 0


def f(w):
    # here we would normally:
    # ... 1) create a neural network with weights w
    # ... 2) run the neural network on the environment for some time
    # ... 3) sum up and return the total reward

    # but for the purposes of an example, lets try to minimize
    # the L2 distance to a specific solution vector. So the highest reward
    # we can achieve is 0, when the vector w is exactly equal to solution
    reward = -np.sum(np.square(solution - w))
    return reward


# hyperparameters
npop = 50  # population size
sigma = 0.1  # noise standard deviation
alpha = 0.001  # learning rate

# start the optimization
solution = np.array([1.5, 1.1, 1.3])
w = np.random.randn(3)  # our initial guess is random
x = 0
y = 0
g = 0
a = 25
while True:

    w = np.random.randn(3)  # our initial guess is random

    j = random.randint(0, 2)
    # print current fitness of the most likely parameter setting

    # initialize memory for a population of w's, and their rewards
    N = np.random.randn(npop, 3)  # samples from a normal distribution N(0,1)
    R = np.zeros(npop)

    w_try = w + sigma * N[j]  # jitter w using gaussian of sigma 0.1
    R[j] = f(w_try)  # evaluate the jittered version

    # standardize the rewards to have a gaussian distribution
    A = (R - np.mean(R)) / np.std(R)
    # perform the parameter update. The matrix multiply below
    # is just an efficient way to sum up all the rows of the noise matrix N,
    # where each row N[j] is weighted by A[j]
    w = w + alpha / (npop * sigma) * np.dot(N.T, A)
    # plt.plot(w)
    # plt.ylabel('numbers')


    x = random.randint(int(abs(w[0] * 50) + 20), int(abs(w[1]) * 200) + 200)
    y = random.randint(int(abs(w[1] * 50) + 120), int(abs(w[2]) * 200) + 400)

    if u > 50:
        action_n = [universe.spaces.PointerEvent(x, y, 1), universe.spaces.PointerEvent(x, y, 0),
                    universe.spaces.PointerEvent(x, y, 1)]

        action_n = [action_n for ob in observation_n]
        observation_n, reward_n, done_n, info = env.step(action_n)

        u = 0
    u += 1

    env.render()