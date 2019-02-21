import numpy as np
from deap import creator, base, tools, algorithms


class GA:

    def __init__(self, ind_size=2, indpb=0.05, tournsize=3, population_size=4, init_individual_func=None,
                 init_individual_args=None, evaluate_func=None):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        self.toolbox = base.Toolbox()
        if init_individual_func is None:
            self.toolbox.register("attr_bool", np.random.randint, 0, 10)
            self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_bool,
                                  n=ind_size)
        else:
            self.toolbox.register("attr_bool", init_individual_func, *init_individual_args)
            self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_bool,
                                  n=ind_size)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        if evaluate_func is None:
            self.toolbox.register("evaluate", self.evalOneMax)
        else:
            self.toolbox.register("evaluate", evaluate_func)
        self.toolbox.register("mate", self.cxTwoPointCopy)
        self.toolbox.register("mutate", tools.mutFlipBit, indpb=indpb)
        self.toolbox.register("select", tools.selTournament, tournsize=tournsize)

        self.pop = self.toolbox.population(n=population_size)
        self.hof = tools.HallOfFame(1, similar=np.array_equal)

        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)

    def evalOneMax(self, individual):
        return sum(individual),

    def cxTwoPointCopy(self, ind1, ind2):
        """Execute a two points crossover with copy on the input individuals. The
        copy is required because the slicing in numpy returns a view of the data,
        which leads to a self overwritting in the swap operation. It prevents
        ::

            >>> import numpy
            >>> a = numpy.array((1,2,3,4))
            >>> b = numpy.array((5.6.7.8))
            >>> a[1:3], b[1:3] = b[1:3], a[1:3]
            >>> print(a)
            [1 6 7 4]
            >>> print(b)
            [5 6 7 8]
        """

        for ind1_i, ind2_i in zip(ind1, ind2):
            size = len(ind1_i)
            print("size= {}".format(size))
            cxpoint1 = np.random.randint(1, size)
            cxpoint2 = np.random.randint(0, size - 1)
            if cxpoint2 >= cxpoint1:
                cxpoint2 += 1
            else:  # Swap the two cx points
                cxpoint1, cxpoint2 = cxpoint2, cxpoint1

            ind1_i[cxpoint1:cxpoint2], ind2_i[cxpoint1:cxpoint2] = ind2_i[cxpoint1:cxpoint2].copy(), \
                                                                   ind1_i[cxpoint1:cxpoint2].copy()

        return ind1_i, ind2_i

    def run_now(self, cxpb=0.5, mutpb=0.2, ngen=40, verbose=True):
        pop, log = algorithms.eaSimple(population=self.pop, toolbox=self.toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen,
                                       stats=self.stats, halloffame=self.hof, verbose=verbose)
        return {'chromosome': list(self.hof.items[0]), 'fitness_score': self.hof.keys[0].wvalues[0], 'pop': pop,
                'log': log}


if __name__ == '__main__':
    x = GA()
    x.run_now()

    print("done")
