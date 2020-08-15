# initialization
# evalutation
# selection
# mutation/crossover

import gameAI
import random
import time
import numpy as np
from deap import creator, base, tools, algorithms

nodes = (11, 8, 6, 3)
count = 0
array_length = 0
for i in range(len(nodes) - 1):
    array_length += nodes[i] * nodes[i+1] + nodes[i+1]


def relu(array):
    for i in range(len(array)):
        array[i] = max(0, array[i])
    return array


def evaluate(array, delay=0, display=False):
    global count
    count += 1
    array = np.array(array)
    game = gameAI.Snake(display)
    state = game.return_state()
    state = np.array(state)
    # parse the array
    weights = []
    biases = []
    start_weight = 0
    start_bias = 0
    for between in range(len(nodes) - 1):
        num_weights = nodes[between] * nodes[between+1]
        num_biases = nodes[between+1]
        start_bias = start_weight + num_weights
        weights.append(array[start_weight:start_weight +
                             num_weights].reshape(nodes[between], nodes[between+1]))
        biases.append(array[start_bias:start_bias+num_biases])
        start_weight += num_weights + num_biases
    reward = 0

    while not game.done:
        time.sleep(delay)
        # Determine reward value of previous step
        if state[7]:
            reward += 1
        if state[6]:
            reward += .10
        else:
            reward += -.15

        if (reward < -5):
            game.done = True

        for i in range(len(nodes)-1):
            state = relu(np.dot(state, weights[i]) + biases[i])
        action = state.argmax() - 1
        state = game.step(action)
    print(count, ": ", reward)
    return (reward,)

# array[48 weights (numbers), 6 biases, 18 weights, 3 biases]


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("attr_flt", random.uniform, -1.0, 1.0)
toolbox.register("individual", tools.initRepeat,
                 creator.Individual, toolbox.attr_flt, n=array_length)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.2, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=5)
#toolbox.register("select", tools.selBest)

population = toolbox.population(n=100)

hof = tools.HallOfFame(1)

final_population = algorithms.eaSimple(
    population, toolbox, cxpb=0.30, mutpb=0.20, ngen=100, verbose=False, halloffame=hof)

print("-------------------------------------------")
print(hof[0])
print("Best Eval:")
print("-------------------------------------------")
print(evaluate(hof[0], .1, True))

# 8 input nodes
# (weights) + biases
# 6 nodes
# (weights) + biases
# 3 output nodes
#     left
#     forward
#     right

# Layer 1 - 48 weights + 6 biases
# Layer 2 - 18 weights + 3 biases
# 75 items in the array in total
