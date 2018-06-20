import csv
import numpy as np
import matplotlib.pyplot as plt

def plotter(x):
    plt.ylabel('Current average reward')
    plt.xlabel('Number of episodes')
    plt.title('Progress with SD 1.0')
    plt.plot(x, label='Test', color="red")


rewardListLow = []
rewardListHigh = []
rewardListWindow = []
sumRewardWindow = 0
sumReward=0
graphTrainLow = np.loadtxt("TrainLR00005VectorDeviation10.txt")
graphNoTrain = np.loadtxt("NoTrain/noTrainCurrentRewardDeviation10.txt")
graphTrainHigh = np.loadtxt("Train/currentRewardDeviation10.txt")


for x in range(1, len(graphTrainLow)):
    sumReward = sumReward + graphTrainLow[x-1]
    rewardListLow.append(sumReward/x)

sumReward = 0
for x in range(1, len(graphNoTrain)):
    sumReward = sumReward + graphNoTrain[x-1]
    rewardListWindow.append(sumReward/x)

sumReward = 0
for x in range(1, len(graphTrainHigh)):
    sumReward = sumReward + graphTrainHigh[x-1]
    rewardListHigh.append(sumReward/x)


plotter(rewardListWindow)
plt.plot(rewardListLow, label='Training with LR = 0.0005', color="green")
plt.plot(rewardListHigh, label='Training with LR = 0.001', color="lime")
plt.legend()
plt.savefig('trainLR00005VSNoTrain10.png')

plt.show()



