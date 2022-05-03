import numpy
import numpy as np
import random
import matplotlib.pyplot as plt

mu = 0.5  # Convergence parameter (range 0.1 to 0.5)
max_iter = 10000000  # Number of iterations (Time count)
N = 1000  # Number of agents
lat_accept = 0.2  # latitude of acceptance by agent
lat_reject = 1.2  # latitude of rejection by agent
cult = 20  # percentage of cultural exchange
f_size = 5  # size of a single feature
number_of_traits = 9  # number of traits per feature (range from 0 to 9)


def main():
    global agent, agenti, features
    agent = [random.random() for i in range(0, N)]  # assigning between 0 and 1 for agents opinions
    features = np.random.randint(number_of_traits, size=(N, f_size))


# agent is the array to store the agents opinions, while agent[ai] would mean agent i's opinion
def share():
    global agent, agenti, features, similarity
    for randomagent in range(N):
        i = random.randint(0, N - 1)
        j = random.randint(0, N - 1)
        if i != j:
            break

    diff = abs(agent[i] - agent[j])
    opi1 = agent[i]
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
        if diff <= lat_accept:
            agent[i] = (mu * agent[j]) + ((1 - mu) * agent[i])
            agent[j] = (mu * opi1) + ((1 - mu) * agent[j])
        elif diff > lat_reject:
            agent[i] = ((1 + mu) * agent[i]) - (mu * agent[j])
            agent[j] = ((1 + mu) * agent[j]) - (mu * opi1)

    elif diff > lat_reject:
        agent[i] = ((1 + mu) * agent[i]) - (mu * agent[j])
        agent[j] = ((1 + mu) * agent[j]) - (mu * opi1)


def Extremists():  #
    TotExtremists = 0
    for i in range(len(agent)):
        if (agent[i]) >= 0.9 or (agent[i]) <= 0.1:
            TotExtremists += 1
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
    for i in range(len(agent)):
        cc = []
        if agent[i] not in km:
            cc.append(agent[i])
            km.append(agent[i])
        for j in range(i + 1, N):
            if agent[j] not in km:
                d = numpy.linalg.norm(agent[i] - agent[j])
                if d < epsilon:
                    cc.append(agent[j])
                    km.append(agent[j])
        cm.append(cc)

    for l in cm:
        if len(l) >= 3:
            ctr += 1
    print("Number of Clusters = ", ctr)

    biggest_Cluster = max(len(u) for u in cm)  # using max
    print("Agents in the biggest cluster = ", biggest_Cluster)


def alliter():  # iteration Vs opinion diffusion graph
    agenti = []
    iteration = np.arange(1, max_iter + 1)
    for it in range(max_iter):
        share()
        for value in agent:
            agenti.append(value)

    cluster_Size = N
    Split = lambda agenti, x: [agenti[i:i + x] for i in range(0, len(agenti),
                                                              x)]  # 1.1 Got idea for this from https://www.delftstack.com/howto/python/python-split-list-into-chunks/#:~:text=of%20n%20objects.-,Split%20List%20in%20Python%20to%20Chunks%20Using%20the%20lambda%20Function,it%20into%20N%2Dsized%20chunks.
    diffusion = Split(agenti,
                      cluster_Size)  # 1.2 Got idea for this from https://www.delftstack.com/howto/python/python-split-list-into-chunks/#:~:text=of%20n%20objects.-,Split%20List%20in%20Python%20to%20Chunks%20Using%20the%20lambda%20Function,it%20into%20N%2Dsized%20chunks.

    plt.plot(iteration, diffusion, "b.")
    plt.xlabel('Iteration', fontsize=10)
    plt.ylabel('Opinions', fontsize=10)
    plt.show()


def diffusion():  # Final Diffused opinion Vs Agents Graph
    AllAgents = np.arange(1, N + 1)
    for it in range(max_iter):
        share()

    plt.plot(AllAgents, agent, 'bo')
    plt.xlabel('Agents', fontsize=10)
    plt.ylabel('Opinion', fontsize=10)
    plt.show()


main()
# We can use the functions alone and make experiments,
# you can uncomment only one function and use it to run that specific part of the model
# (use one function at a time, running two might not give accurate results).
"""
alliter()

cluster()
extrePlot()
"""

diffusion()
