# %% import libraries
import time
import random
import numpy as np

from sklearn.linear_model import LinearRegression


# %% genetic algorithm
# class representing individual in population
class Individual:
    # initialization
    def __init__(self, X, T, reg_method, max_depth=None, chromosome=None):
        self.X = X.copy()
        self.T = T.copy()
        self.reg_method = reg_method
        self.max_depth = max_depth
        if chromosome is None:
            self.chromosome = self.create_gnome()
        else:
            self.chromosome = chromosome
        self.fitness_train, self.fitness_test = self.cal_fitness(X[:, np.array(self.chromosome) == 1], T)

    @staticmethod
    def mutated_genes():
        # create random genes for mutation
        return random.randint(0, 1)

    def create_gnome(self):
        # create chromosome or string of genes
        return [random.randint(0, 1) for _ in range(self.X.shape[1])]

    def mate(self, par2):
        # perform mating and produce new offspring

        # chromosome for offspring
        child_chromosome = []
        for gp1, gp2 in zip(self.chromosome, par2.chromosome):

            # random probability
            prob = random.random()

            # if prob is less than 0.45, insert gene from parent 1
            if prob < 0.45:
                child_chromosome.append(gp1)

            # if prob is between 0.45 and 0.90, insert gene from parent 2
            elif prob < 0.90:
                child_chromosome.append(gp2)

            # otherwise insert random gene(mutate) for maintaining diversity
            else:
                child_chromosome.append(self.mutated_genes())

        # create new Individual(offspring) using
        # generated chromosome for offspring
        return Individual(self.X, self.T, self.reg_method, self.max_depth, chromosome=child_chromosome)

    def cal_fitness(self, X, T):
        # calculate mape of current selected features
        if self.max_depth is None:
            result_train, result_test = self.reg_method(X, T)
        else:
            result_train, result_test = self.reg_method(X, T, self.max_depth)

        return result_train, result_test


def GeneticAlgorithm(X, T, max_gen=30, pop_size=10, **params):
    time_start = time.time()
    reg_method = params.pop('reg_method', LinearRegression)
    max_depth = params.pop('max_depth', None)
    population = []

    print('----- ----- Genetic Algorithm Starts ----- -----')
    # create initial population
    for _ in range(pop_size):
        population.append(Individual(X, T, reg_method, max_depth))

    for cur_gen in range(max_gen):
        # sort the population in increasing order of fitness score
        population = sorted(population, key=lambda x: x.fitness_train)

        # generate new offsprings for new generation
        new_generation = []

        # perform Elitism, that mean 10% of fittest population goes to the next generation
        s = int((10 * pop_size) / 100)
        new_generation.extend(population[:s])

        # from 90% of fittest population, Individuals will mate to produce offspring
        s = int((90 * pop_size) / 100)
        for _ in range(s):
            parent1 = random.choice(population[:50])
            parent2 = random.choice(population[:50])
            child = parent1.mate(parent2)
            new_generation.append(child)

        population = new_generation

        # extract validation MAPE
        fitness = population[0].fitness_test

        print(f"Generation: {cur_gen} | Number of Features: {sum(population[0].chromosome)} | Training RMSE: {population[0].fitness_train} | Validation RMSE: {fitness}")

        # record search history
        if cur_gen == 0:
            search_history = np.array([cur_gen, sum(population[0].chromosome), population[0].fitness_train, fitness])
        else:
            search_history = np.vstack([search_history, np.array([cur_gen,
                                                                  sum(population[0].chromosome),
                                                                  population[0].fitness_train,
                                                                  fitness])])
    print(f"- Training Time Elapsed: {int((time.time() - time_start) / 3600):.0f}h "
          f"{int(((time.time() - time_start) / 60) % 60):.0f}m "
          f"{int((time.time() - time_start) % 60):.0f}s -")
    return search_history, population[0]