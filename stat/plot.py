import os
import matplotlib.pyplot as plt
import numpy as np
from heapq import *


# type what you want to display here
x_axis = ['x', 'runtime']
# metric_types = ["replayBuffStats", "episodeStepRewards", "episodeAverages", "trainingLoss"]
metric_types = ["episodeStepRewards", "episodeAverages", "trainingLoss"]
metric_names = [
    # ["runtime", "x", "startingRange", "currentBufferSize"],
    ["runtime", "x", "Step_Reward", "EMA_Step_Rewards", "Step_Obj_Pred_Error", "EMA_Obj_Pred_Error"],
    ["runtime", "x", "episodeNumber", "EpisodeAverageRewardsPerStep", "TotalEpisodeSteps", "EMAEpisodeStepRewards", "EMATotalEpisodeSteps", "EMAEpisodeObjPredError"],
    ["runtime", "x", "policy_loss", "value_loss", "q_loss_1", "q_loss_2", "_alpha"]
]


for names, dimension in zip(metric_names, metric_types):
    data = np.genfromtxt(os.path.join(os.getcwd(), "stat/v4/" + dimension + ".txt"), delimiter=',', dtype=None, names=names)
    data.sort(order=["runtime"])

    # 'runtime' and 'x' should not be plotted
    counters = 0
    for counter in x_axis:
        if counter in data.dtype.names:
            counters += 1
    
    # Ready figure with 'n' subplots
    fig, axs = plt.subplots(len(names) - counters)
    
    # Plot all data
    index = 0
    for name in names:
        if name != "runtime" and name != "x":
            axs[index].plot(data['x'], data[name])
            # axs[index].plot(data['runtime'], data[name])

            axs[index].set_title(name)
            index += 1

    plt.show()