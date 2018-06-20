import csv
import numpy as np
import matplotlib.pyplot as plt

graphNoWindow = np.loadtxt("currentReward.txt")
graphWithWindow = np.loadtxt("windowReward.txt")
print(graphWithWindow)
# print(graphNoWindow)

# graphWithWindow = np.repeat(graphWithWindow, 5)
# plt.plot(graphNoWindow)
plt.plot(graphNoWindow)
plt.ylabel('some numbers')
plt.show()

def plotter(x):
    plt.ylabel('Current average reward')
    plt.xlabel('Number of episodes')
    plt.title('Progress of RL')
    plt.plot(x)
    plt.savefig('ProgressRL2.png')

def plotWindow(y):
    plt.ylabel('Average reward over one window')
    plt.xlabel('Number of timewindows')
    plt.title('Progress of RL with timewindow = 500')
    plt.plot(y)
    plt.savefig('ProgressRLWindows2.png')
