import os
import matplotlib.pyplot as plt
import numpy as np

from numpy import arange,array,ones
import scipy
from scipy.optimize import curve_fit
from scipy import stats
from heapq import *


# type what you want to display here
line_types = ["linear", "log"]
line_type = line_types[0]
dimensions = ["episodeAverages", "trainingLoss"]
dimension = dimensions[0]

# load in data and sort
heap = []
# runtime, x, numEpisodes, episodeRewards, episodeSteps, episodeAverageRewards, episodeAverageSteps = np.loadtxt(dimension + ".txt", delimiter=',', unpack=True)
# runtime, x, policy_loss, value_loss, q_loss_1, q_loss_2 = np.loadtxt(dimension + ".txt", delimiter=',', unpack=True)
# runtime, x, _, _, _, y = np.loadtxt(os.path.join(os.getcwd(), "stat", dimension + ".txt"), delimiter=',', unpack=True)
runtime, x, y, _, _, _ = np.loadtxt(os.path.join(os.getcwd(), "stat", dimension + ".txt"), delimiter=',', unpack=True)
for i in range(len(runtime)):
    data = abs(y[i])
    time = runtime[i]
    item = (time, data)
    heappush(heap, item)
  
sorted_x = []
sorted_y = []
counter = 0

while heap:
    (x_i, y_i) = heappop(heap)
    sorted_x.append(counter)
    sorted_y.append(y_i)
    counter += 1

sorted_x = np.array(sorted_x, dtype=int)

if line_type == 'linear':
    
    # Best Fit line
    slope, intercept, r_value, p_value, std_err = stats.linregress(sorted_x, sorted_y)
    line = slope * sorted_x + intercept
    plt.plot(sorted_x, sorted_y, label='Slope: ' + str(slope))

    # plt.plot(sorted_x, sorted_y,'o', sorted_x, line)

    plt.xlabel('Time')
    plt.ylabel(dimension)
    plt.title('Statistics View')
    plt.legend()
    plt.show()

elif line_type == 'log':
    # [scale, offset] = np.polyfit(np.log(sorted_x), sorted_y, 1)
    [[offset, scale], _ ] = scipy.optimize.curve_fit(lambda t,a,b: a+b*np.log(t),  x,  y)

    line = scale * np.log(x) + offset

    plt.plot(sorted_x, sorted_y, label='Fitted Line: ' + str(scale) + " log(x)" + " + " + str(offset))

    plt.plot(sorted_x, sorted_y,'o', sorted_x, line)
    plt.plot(sorted_x, sorted_y,'o')

    plt.xlabel('Time')
    plt.ylabel(dimension)
    plt.title('Statistics View')
    plt.legend()
    plt.show()
    # x = sorted_x
    # y = sorted_y

    # def func(x, a, b, c):
    #     #return a * np.exp(-b * x) + c
    #     return a * np.log(b * x) + c

    # line = y + 0.2*np.random.normal(size=len(x))

    # popt, pcov = curve_fit(func, x, line)

    # plt.figure()
    # plt.plot(x, line, 'ko', label="Original Noised Data")
    # plt.plot(x, func(x, *popt), 'r-', label="Fitted Curve")
    # plt.legend()
    # plt.show()
