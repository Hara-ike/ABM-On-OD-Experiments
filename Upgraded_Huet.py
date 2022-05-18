import numpy
import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx

mu = 0.5  # Kinetic Parameter representing velocity of rejection
U_same = 0.25  # Uncertainty for opinions in same group
U_Diff = 0.15  # Uncertainty for opinions in diff group
delta = 1  # confidence threshold for uncertainty
max_iter = 10000000  # Number of iterations (Time count)
N = 1000  # Number of agents
c = 50  # maximum number of communities in the model

def ps(s):
    New_attitude = 0
    if s < 0:
        New_attitude = -1
    else:
        New_attitude = 1
    return New_attitude

# op1 = opinion 1 and op2 = opinion 2, incase op1[i], it would be opinion 1 of agent i

def main():
    global op1, op2, agenti, community
    op1 = [random.uniform(-1, 1) for i in range(0, N)]  # assigning between -1 and 1 for opinions
    op2 = [random.uniform(-1, 1) for i in range(0, N)]  # assigning between -1 and 1 for opinions
    community = [random.randint(1, c) for i in range(N)]  # creating a community of agents (Environment of near agents)


def share():  # the part where interaction occurs and the opinions are updated
    global op1, op2, agenti

    for randomagent in range(N):
        i = random.randint(0, N - 1)
        j = random.randint(0, N - 1)
        if i != j:
            break

    d1 = abs(op1[i] - op1[j])
    d2 = abs(op2[i] - op2[j])
    if community[i] == community[j]:
        if (d1 <= U_same) and (d2 <= U_same):
            op1[i] = op1[i] + (mu * (op1[j] - op1[i]))
            op1[j] = op1[j] + (mu * (op1[i] - op1[j]))
            op2[i] = op2[i] + (mu * (op2[j] - op2[i]))
            op2[j] = op2[j] + (mu * (op2[i] - op2[j]))

        elif (d1 <= U_same) and (d2 > U_same):
            if d2 <= ((1 + delta) * U_same):
                op1[i] = op1[i] + (mu * (op1[j] - op1[i]))
                op1[j] = op1[j] + (mu * (op1[i] - op1[j]))
            else:
                op1[i] = op1[i] - (mu * ps(op1[j] - op1[i])) * (U_same - (abs(op1[j] - op1[i])))
                op1[j] = op1[j] - (mu * ps(op1[i] - op1[j])) * (U_same - (abs(op1[i] - op1[j])))

        elif (d2 <= U_same) and (d1 > U_same):
            if d1 <= ((1 + delta) * U_same):
                op2[i] = op2[i] + (mu * (op2[j] - op2[i]))
                op2[j] = op2[j] + (mu * (op2[i] - op2[j]))
            else:
                op2[i] = op2[i] - (mu * ps(op2[j] - op2[i])) * (U_same - (abs(op2[j] - op2[i])))
                op2[j] = op2[j] - (mu * ps(op2[i] - op2[j])) * (U_same - (abs(op2[i] - op2[j])))
    else:
        if (d1 <= U_Diff) and (d2 <= U_Diff):
            op1[i] = op1[i] + (mu * (op1[j] - op1[i]))
            op1[j] = op1[j] + (mu * (op1[i] - op1[j]))
            op2[i] = op2[i] + (mu * (op2[j] - op2[i]))
            op2[j] = op2[j] + (mu * (op2[i] - op2[j]))

        elif (d1 <= U_Diff) and (d2 > U_Diff):
            if d2 <= ((1 + delta) * U_Diff):
                op1[i] = op1[i] + (mu * (op1[j] - op1[i]))
                op1[j] = op1[j] + (mu * (op1[i] - op1[j]))
            else:
                op1[i] = op1[i] - (mu * ps(op1[j] - op1[i])) * (U_Diff - (abs(op1[j] - op1[i])))
                op1[j] = op1[j] - (mu * ps(op1[i] - op1[j])) * (U_Diff - (abs(op1[i] - op1[j])))

        elif (d2 <= U_Diff) and (d1 > U_Diff):
            if d1 <= ((1 + delta) * U_Diff):
                op2[i] = op2[i] + (mu * (op2[j] - op2[i]))
                op2[j] = op2[j] + (mu * (op2[i] - op2[j]))
            else:
                op2[i] = op2[i] - (mu * ps(op2[j] - op2[i])) * (U_Diff - (abs(op2[j] - op2[i])))
                op2[j] = op2[j] - (mu * ps(op2[i] - op2[j])) * (U_Diff - (abs(op2[i] - op2[j])))

    if abs(op1[i]) > 1: op1[i] = ps(op1[i])
    if abs(op1[j]) > 1: op1[j] = ps(op1[j])
    if abs(op2[i]) > 1: op2[i] = ps(op2[i])
    if abs(op2[j]) > 1: op2[j] = ps(op2[j])


def Extremists():  # algorithm to identify Extremists in the system
    TotExtremists = 0
    for i in range(len(op1)):
        if abs(op1[i]) >= 0.9 or abs(op2[i]) >= 0.9:
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
    for i in range(len(op1)):
        cc = []
        if op1[i] not in km:
            cc.append(op1[i])
            km.append(op1[i])
        for j in range(i + 1, N):
            if op1[j] not in km:
                d = numpy.linalg.norm(op1[i] - op1[j])
                if d < epsilon:
                    cc.append(op1[j])
                    km.append(op1[j])
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
        for value in op2:
            agenti.append(value)

    cluster_Size = N
    Split = lambda agenti, x: [agenti[i:i + x] for i in range(0, len(agenti),
                                                              x)]  # 1.1 Got idea for this from https://www.delftstack.com/howto/python/python-split-list-into-chunks/#:~:text=of%20n%20objects.-,Split%20List%20in%20Python%20to%20Chunks%20Using%20the%20lambda%20Function,it%20into%20N%2Dsized%20chunks.
    diffusion = Split(agenti,
                      cluster_Size)  # 1.2 Got idea for this from https://www.delftstack.com/howto/python/python-split-list-into-chunks/#:~:text=of%20n%20objects.-,Split%20List%20in%20Python%20to%20Chunks%20Using%20the%20lambda%20Function,it%20into%20N%2Dsized%20chunks.

    plt.plot(iteration, diffusion, "b.")
    plt.xlabel('Iteration', fontsize=10)
    plt.ylabel('Attitude 2', fontsize=10)
    plt.savefig('5.png')


def diffusion():  # Final Diffused opinion Vs Agents Graph
    AllAgents = np.arange(1, N + 1)
    for it in range(max_iter):
        share()

    plt.plot(AllAgents, op1, 'bo')
    plt.xlabel('Agents', fontsize=10)
    plt.ylabel('Opinion 1', fontsize=10)
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
