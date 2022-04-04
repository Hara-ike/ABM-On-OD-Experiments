import numpy
import numpy as np
import random
import matplotlib.pyplot as plt

miu = 0.5  # Convergence paramter (range 0.1 to 0.5)
d = 0.25  # threshold
max_iter = 28000  # Number of iterations (Time count)
N = 1000  # Number of agents


def main():
    global agent, agenti
    agent = [random.random() for i in range(0, N)]  # assigning either 0 or 1 for agents opinions


def share():
    global agent, agenti
    while True:
        ai = random.randint(0, N - 1)
        aj = random.randint(0, N - 1)
        if ai != aj:
            break
    diff = abs(agent[ai] - agent[aj])
    if d > diff:
        agent[ai] = agent[ai] + miu * (agent[aj] - agent[ai])
        agent[aj] = agent[aj] + miu * (agent[ai] - agent[aj])


def Extremists():  #
    TotExtremists = 0
    Extre = []
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


def alliter():  # iteration Vs opinion diffusion graph
    agenti = []
    iteration = np.arange(1, max_iter + 1)
    for it in range(max_iter):
        share()
        for value in agent:
            agenti.append(value)

    x = N
    final_list = lambda agenti, x: [agenti[i:i + x] for i in range(0, len(agenti),x)]  # 1.1 Got idea for this from https://www.delftstack.com/howto/python/python-split-list-into-chunks/#:~:text=of%20n%20objects.-,Split%20List%20in%20Python%20to%20Chunks%20Using%20the%20lambda%20Function,it%20into%20N%2Dsized%20chunks.
    output = final_list(agenti,x)  # 1.2 Got idea for this from https://www.delftstack.com/howto/python/python-split-list-into-chunks/#:~:text=of%20n%20objects.-,Split%20List%20in%20Python%20to%20Chunks%20Using%20the%20lambda%20Function,it%20into%20N%2Dsized%20chunks.
    plt.plot(iteration, output, color="blue")
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


main()  # plotting starts here, we can use the functions alone and make experiments
"""
alliter()
"""
"""
diffusion()
"""
"""
extrePlot()
"""


"""   
plt.plot(kio,output,color="Blue")
plt.savefig('t.png')
def cluster():
    cm = []
    epsilon = 0.05
    ctr = 0
    for i in range(len(agent)):
        cc = []
        if agent[i] not in cm:
            cc.append(agent[i])
            ctr.append(i)
        for j in range(i+1,N):
            d = numpy.linalg.norm(agent[i]-agent[j])
            if d < epsilon:
                cc.append(agent[j])
                ctr = ctr +1
        cm.append(cc)
    np.savetxt('jjj.csv',cm,fmt='%s')
"""
