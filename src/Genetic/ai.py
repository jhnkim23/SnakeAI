#initialization
#evalutation
#selection
#mutation/crossover

import gameAI
import random
import numpy as np
from deap import creator, base, tools, algorithms

nodes = (8, 6, 3)

def relu(array):
    for i in range(len(array)):
        array[i] = max(0, array[i])
    return array

def evaluate(array):
    array = np.array(array)
    game = gameAI.Snake()
    state = game.return_state()
    state = np.array(state)
    #parse the array
    weights = []
    biases = []
    start_weight = 0
    start_bias = 0
    for between in range(len(nodes) - 1):
        num_weights = nodes[between] * nodes[between+1]
        num_biases = nodes[between+1]
        start_bias = num_weights
        weights.append(array[start_weight:start_weight+num_weights].reshape(nodes[between], nodes[between+1]))
        biases.append(array[start_bias:start_bias+num_biases])
        start_weight += num_weights + num_biases    
    reward = 0

    while not game.done:
        #Determine reward value of previous step
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
    print(reward)
    return (reward,)

# array[48 weights (numbers), 6 biases, 18 weights, 3 biases]


creator.create("FitnessMax", base.Fitness, weights = (1.0,))
creator.create("Individual", list, fitness = creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("attr_flt", random.uniform, -1.0, 1.0)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_flt, n = 75)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.2, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize = 5)

population = toolbox.population(n = 100)

final_population = algorithms.eaSimple(population, toolbox, cxpb=0.4, mutpb=0.3, ngen=500, verbose=False)

best = tools.selBest(final_population, k=1)
print(evaluate(best))

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