import numpy as np
import matplotlib.pyplot as plt

# array = np.linspace(-1,1,50)
# np.savetxt('arraySimpleChooseButton.txt', array, fmt='%.2f', delimiter=',',
# 								   header="Rewards on eacht moment!")

def plotter(x):
    plt.ylabel('Current average reward')
    plt.xlabel('Number of episodes')
    plt.title('Progress in machine learning task')
    plt.plot(x,  color="red")


rewardListLow = []
rewardListHigh = []
rewardListWindow = []
sumRewardWindow = 0
sumReward=0
graphTrainLow = np.loadtxt("arraySimpleChooseButton.txt")
# graphNoTrain = np.loadtxt("NoTrain/noTrainCurrentRewardDeviation10.txt")
# graphTrainHigh = np.loadtxt("Train/currentRewardDeviation10.txt")
print(graphTrainLow)

for x in range(1, len(graphTrainLow)):
    sumReward = sumReward + graphTrainLow[x-1]
    rewardListLow.append(sumReward/x)

# sumReward = 0
# for x in range(1, len(graphNoTrain)):
#     sumReward = sumReward + graphNoTrain[x-1]
#     rewardListWindow.append(sumReward/x)
#
# sumReward = 0
# for x in range(1, len(graphTrainHigh)):
#     sumReward = sumReward + graphTrainHigh[x-1]
#     rewardListHigh.append(sumReward/x)


plotter(rewardListLow)
# plt.plot(rewardListLow, label='Training with LR = 0.0005', color="green")
# plt.plot(rewardListHigh, label='Training with LR = 0.001', color="lime")
# plt.legend()
plt.savefig('simpleChooseButton.png')

plt.show()