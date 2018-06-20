import csv
import numpy as np
import matplotlib.pyplot as plt
import math

# def plotter(x):
#     plt.ylabel('Current average reward')
#     plt.xlabel('Number of episodes')
#     plt.title('Progress with SD 1')
#     plt.plot(x, label='Test')
#
#
# rewardList = []
# rewardListWindow = []
# sumRewardWindow = 0
# sumReward=0
# graph = np.random.logseries(.6, 1000)
#
# for i in range(0,graph.size):
#     graph[i] = graph[i] + i*.001 -2
#
# for i in range(0,200):
#     graph[i] = graph[i] + i*.001
# # graph = - np.log(graph)
# print(graph)
#
# a = np.random
#
# for i in range (0,1000):
#     a[i] = i
# X = np.linspace(0, 1000, 100)
# Y = (2*X) + 2 + 20*np.random.randn(100) * a
# data = np.hstack((X.reshape(100,1),Y.reshape(100,1)))
# plotter(Y)
#
# # for exponent in range(0,graph.size):
# #     graph[exponent] =  graph[exponent]  * np.log()
# # graph = np.random.exponential(1,(1000))
# # graphNoTrain = np.loadtxt("NoTrain/noTrainCurrentRewardDeviation10.txt")
#
# for x in range(1, len(Y)):
#     sumReward = sumReward + Y[x-1]
#     rewardList.append(sumReward/x)
#
#
# sumReward = 0
# # for x in range(1, len(graphNoTrain)):
# #     sumReward = sumReward + graphNoTrain[x-1]
# #     rewardListWindow.append(sumReward/x)
#
# # plotter(rewardListWindow)
# plt.plot(rewardList, label='With train')
# plt.legend()
# # plt.savefig('TrainVsNoTrain10.png')
#
# plt.show()
#

x = np.linspace(0, 10, 1000)
y = np.exp(1*x) + np.random.randn(1000)

plt.figure()
plt.plot(x, -np.exp(-x))
plt.xlabel('$x$')
plt.ylabel('$-\exp(-x)$')

plt.show()

