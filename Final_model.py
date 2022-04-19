import numpy
import numpy as np
import random
import matplotlib.pyplot as plt


mu = 0.5  # Kinetic Parameter representing velocity of rejection and attraction
U_same = 0.25  # Uncertainty for opinion in same neighbourhood
U_diff = 0.15  # Uncertainty for opinion in different neighbourhood
delta = 1  # confidence threshold for uncertainty
max_iter = 10000000  # Number of iterations (Time count)
N = 1000  # Number of agents
c = 50  # Number of neighbourhoods present in a community
lat_accept = 0.3  # latitude of acceptance by agent i for same neighbourhoods
lat_reject = 1.2  # latitude of rejection by agent i for same neighbourhoods
cult = 20  # percentage of cultural exchange
f_size = 5  # size of a single feature
number_of_traits = 4  # number of traits per feature (range from 0 to 4)


def ps(s):
    New_attitude = 0
    if s < 0:
        New_attitude = -1
    else:
        New_attitude = 1
    return New_attitude


# op1 = opinion 1 and op2 = opinion 2, incase op1[i], it would be opinion 1 of agent i

def main():
    global op1, op2, opi1, opi2, opj1, opj2, agenti, community, features
    op1 = [random.uniform(-1, 1) for i in range(0, N)]  # assigning between -1 and 1 for attitudes
    op2 = [random.uniform(-1, 1) for i in range(0, N)]  # assigning between -1 and 1 for attitudes
    community = [random.randint(1, c) for i in range(N)]  # creating a community of agents (Environment of near agents)
    features = np.random.randint(number_of_traits, size=(N, f_size))


def share():  # the part where interaction occurs and the attitudes are updated
    global op1, op2, opi1, opi2, opj1, opj2, agenti, community, features, similarity

    for randomagent in range(N):
        i = random.randint(0, N - 1)
        j = random.randint(0, N - 1)
        if i != j:
            break

    d1 = abs(op1[i] - op1[j])
    d2 = abs(op2[i] - op2[j])
    opi1 = op1[i]
    opi2 = op2[i]
    opj1 = op1[j]
    opj2 = op2[j]

    feature_i = features[i]
    feature_j = features[j]

    similarity = 0
    for trait in range(f_size):
        if feature_i[trait] == feature_j[trait]:
            similarity = similarity + 1
    similarity = similarity * (1 / f_size) * 100

    if similarity >= cult:
        for change in range(f_size):
            if feature_i[change] != feature_j[change]:
                feature_i[change] = feature_j[change]
                break
        if community[i] == community[j]:
            if (d1 <= U_same) and (d2 <= U_same) and (d1 < lat_accept) and (d2 < lat_accept):
                op1[i] = (mu * op1[j]) + ((1 - mu) * op1[i])
                op1[j] = (mu * opi1) + ((1 - mu) * op1[j])
                op2[i] = (mu * op2[j]) + ((1 - mu) * op2[i])
                op2[j] = (mu * opi2) + ((1 - mu) * op2[j])

            elif (d1 > U_same) and (d1 > U_same) and (d1 > lat_reject) and (d2 > lat_reject):
                op1[i] = ((1 + mu) * op1[i]) - (mu * op1[j])
                op1[j] = ((1 + mu) * op1[j]) - (mu * opj1)
                op2[i] = ((1 + mu) * op2[i]) - (mu * op2[j])
                op2[j] = ((1 + mu) * op2[j]) - (mu * opj2)

            elif (d1 <= U_same) and (d2 > U_same):
                if d2 <= ((1 + delta) * U_same):
                    op1[i] = op1[i] + (mu * (op1[j] - op1[i]))
                    op1[j] = op1[j] + (mu * (opi1 - op1[j]))
                else:
                    op1[i] = op1[i] - (mu * ps(op1[j] - op1[i])) * (U_same - (abs(opj1 - op1[i])))
                    op1[j] = op1[j] - (mu * ps(op1[i] - op1[j])) * (U_same - (abs(opi1 - op1[j])))

            elif (d2 <= U_same) and (d1 > U_same):
                if d1 <= ((1 + delta) * U_same):
                    op2[i] = op2[i] + (mu * (op2[j] - op2[i]))
                    op2[j] = op2[j] + (mu * (opi2 - op2[j]))
                else:
                    op2[i] = op2[i] - (mu * ps(op2[j] - op2[i])) * (U_same - (abs(opj2 - op2[i])))
                    op2[j] = op2[j] - (mu * ps(op2[i] - op2[j])) * (U_same - (abs(opi2 - op2[j])))
        else:
            if (d1 <= U_diff) and (d2 <= U_diff) and (d1 < lat_accept) and (d2 < lat_accept):
                op1[i] = (mu * op1[j]) + ((1 - mu) * op1[i])
                op1[j] = (mu * opi1) + ((1 - mu) * op1[j])
                op2[i] = (mu * op2[j]) + ((1 - mu) * op2[i])
                op2[j] = (mu * opi2) + ((1 - mu) * op2[j])

            elif (d1 <= U_diff) and (d2 > U_diff):
                if d2 <= ((1 + delta) * U_diff) and (d1 > lat_reject):
                    op1[i] = ((1 + mu) * op1[i]) - (mu * op1[j])
                    op1[j] = ((1 + mu) * op1[j]) - (mu * opj1)
                else:
                    op1[i] = op1[i] - (mu * ps(op1[j] - op1[i])) * (U_diff - (abs(opj1 - op1[i])))
                    op1[j] = op1[j] - (mu * ps(op1[i] - op1[j])) * (U_diff - (abs(opi1 - op1[j])))

            elif (d2 <= U_diff) and (d1 > U_diff):
                if d1 <= ((1 + delta) * U_diff) and (d2 > lat_reject):
                    op2[i] = ((1 + mu) * op2[i]) - (mu * op2[j])
                    op2[j] = ((1 + mu) * op2[j]) - (mu * opj2)
                else:
                    op2[i] = op2[i] - (mu * ps(op2[j] - op2[i])) * (U_diff - (abs(opj2 - op2[i])))
                    op2[j] = op2[j] - (mu * ps(op2[i] - op2[j])) * (U_diff - (abs(opi2 - op2[j])))
    elif d1 > lat_reject and d2 > lat_reject:
        op1[i] = ((1 + mu) * op1[i]) - (mu * op1[j])
        op1[j] = ((1 + mu) * op1[j]) - (mu * opj1)
        op2[i] = ((1 + mu) * op2[i]) - (mu * op2[j])
        op2[j] = ((1 + mu) * op2[j]) - (mu * opj2)

    if abs(op1[i]) > 1: op1[i] = opi1
    if abs(op2[i]) > 1: op2[i] = opi2
    if abs(op1[j]) > 1: op1[j] = opj1
    if abs(op2[j]) > 1: op2[j] = opj2


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
    plt.ylabel('Opinion 2', fontsize=10)
    plt.savefig('l.png')


def diffusion():  # Final Diffused opinion Vs Agents Graph
    AllAgents = np.arange(1, N + 1)
    for it in range(max_iter):
        share()

    plt.plot(op2, op1, 'bo')
    plt.xlabel('Agents', fontsize=10)
    plt.ylabel('Opinion 1', fontsize=10)
    plt.show()


main()
# We can use the functions alone and make experiments,
# you can uncomment only one function and use it to run that specific part of the model
# (use one function at a time, running two might not give accurate results).
"""
cluster()
alliter()
extrePlot()
diffusion()
"""

