#initialization
#evalutation
#selection
#mutation/crossover

import random
import numpy
from deap import creator, base, tools, algorithms

creator.create("FitnessMax", base.Fitness, weights = (1.0,))
creator.create("Individual", numpy.ndarray, fitness = creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("attr_flt", random.uniform, -1.0, 1.0)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_flt, n = 75)
toolbox.register("population", tools.initRepeat, numpy.ndarray, toolbox.individual)

print(toolbox.individual())

# 8 input nodes
# (weights)
# 6 nodes + biases
# (weights)
# 3 output nodes + biases
#     left
#     forward
#     right

# Layer 1 - 48 weights + 6 biases (96)
# Layer 2 - 18 weights + 3 biases