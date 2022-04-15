import numpy
import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx

mu = 0.5  # Kinetic Parameter representing velocity of rejection
U = 0.25 # Uncertainty for attitudes
delta = 0  # confidence threshold for uncertainty
max_iter = 100000  # Number of iterations (Time count)
N = 1000  # Number of agents


def ps(s):
    New_attitude = 0
    if s < 0:
        New_attitude = -1
    else:
        New_attitude = 1
    return New_attitude


# at1 = attitude 1 and at2 = attitude 2, incase at1[i], it would be attitude 1 of agent i

def main():
    global at1, at2, agenti
    at1 = [random.uniform(-1, 1) for i in range(0, N)]  # assigning between -1 and 1 for attitudes
    at2 = [random.uniform(-1, 1) for i in range(0, N)]  # assigning between -1 and 1 for attitudes


def share():  # the part where interaction occurs and the attitudes are updated
    global at1, at2, agenti

    for randomagent in range(N):
        i = random.randint(0, N - 1)
        j = random.randint(0, N - 1)
        if i != j:
            break

    d1 = abs(at1[i] - at1[j])
    d2 = abs(at2[i] - at2[j])
    if (d1 <= U) and (d2 <= U):
        at1[i] = at1[i] + (mu * (at1[j] - at1[i]))
        at2[i] = at2[i] + (mu * (at2[j] - at2[i]))

    elif (d1 <= U) and (d2 > U):
        if d2 <= ((1 + delta) * U):
            at1[i] = at1[i] + (mu * (at1[j] - at1[i]))
        else:
            at1[i] = at1[i] - (mu * ps(at1[j] - at1[i])) * (U - (abs(at1[j] - at1[i])))

    elif (d2 <= U) and (d1 > U):
        if d1 <= ((1 + delta) * U):
            at2[i] = at2[i] + (mu * (at2[j] - at2[i]))
        else:
            at2[i] = at2[i] - (mu * ps(at2[j] - at2[i])) * (U - (abs(at2[j] - at2[i])))

    if abs(at1[i]) > 1: at1[i] = ps(at1[i])
    if abs(at2[i]) > 1: at2[i] = ps(at2[i])


def Extremists():  # algorithm to identify Extremists in the system
    TotExtremists = 0
    for i in range(len(at1)):
        if abs(at1[i]) >= 0.9 or abs(at2[i]) >= 0.9:
            TotExtremists = TotExtremists + 1
    return TotExtremists


def extrePlot():  # iteration Vs Extremists plot
    extre = []
    iter = np.arange(1, max_iter + 1)
    for it in range(max_iter):
        share()
        extre.append(Extremists())

    plt.plot(iter, extre, "b-")
    plt.xlabel('Iteration', fontsize=10)
    plt.ylabel('Extremists', fontsize=10)
    plt.show()


def cluster():
    cm = []
    km = []
    iteration = np.arange(1, N + 1)
    epsilon = 0.05
    ctr = 0
    for it in range(max_iter):
        share()
    for i in range(len(at1)):
        cc = []
        if at1[i] not in km:
            cc.append(at1[i])
            km.append(at1[i])
        for j in range(i + 1, N):
            if at1[j] not in km:
                d = numpy.linalg.norm(at1[i] - at1[j])
                if d < epsilon:
                    cc.append(at1[j])
                    km.append(at1[j])
        cm.append(cc)

    for l in cm:
        if len(l) >= 3:
            ctr = ctr + 1
    print("Number of Clusters = ", ctr)

    biggest_Cluster = max(len(u) for u in cm)  # using max
    print("Agents in the biggest cluster = ", biggest_Cluster)


def alliter():  # iteration Vs opinion diffusion graph
    agenti = []
    iteration = np.arange(1, max_iter + 1)
    for it in range(max_iter):
        share()
        for value in at2:
            agenti.append(value)

    cluster_Size = N
    Split = lambda agenti, x: [agenti[i:i + x] for i in range(0, len(agenti),
                                                              x)]  # 1.1 Got idea for this from https://www.delftstack.com/howto/python/python-split-list-into-chunks/#:~:text=of%20n%20objects.-,Split%20List%20in%20Python%20to%20Chunks%20Using%20the%20lambda%20Function,it%20into%20N%2Dsized%20chunks.
    diffusion = Split(agenti,
                      cluster_Size)  # 1.2 Got idea for this from https://www.delftstack.com/howto/python/python-split-list-into-chunks/#:~:text=of%20n%20objects.-,Split%20List%20in%20Python%20to%20Chunks%20Using%20the%20lambda%20Function,it%20into%20N%2Dsized%20chunks.

    plt.plot(iteration, diffusion, "b.")
    plt.xlabel('Iteration', fontsize=10)
    plt.ylabel('Attitude 2', fontsize=10)
    plt.savefig('25.png')


def diffusion():  # Final Diffused opinion Vs Agents Graph
    AllAgents = np.arange(1, N + 1)
    for it in range(max_iter):
        share()

    plt.plot(AllAgents, at2, 'bo')
    plt.xlabel('Agents', fontsize=10)
    plt.ylabel('Attribute 1', fontsize=10)
    plt.show()


main()
# We can use the functions alone and make experiments,
# you can uncomment only one function and use it to run that specific part of the model
# (use one function at a time, running two might not give accurate results).
"""
alliter()
extrePlot()
diffusion()
cluster()
"""

