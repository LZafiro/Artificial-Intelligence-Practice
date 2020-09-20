# Particle Swarm Optimization (PSO)
# Luiz Felipe Raveduti Zafiro - 120513
# Artifitial Intelligence - Computer Engineering 
# Federal University of São Paulo (SJC) - 2020

import random
import numpy as np

"""
-> PSO function executes the PSO algorithm for a given function to be evaluated

-> Equations
####################################################################
v(i + 1) = w*v(i) + c1*r1*(Pbest - x(i)) + c2*r2*(Gbest - x(i))
x(i + 1) = x(i) + v(i + 1)
####################################################################
"""

# Class definition of the PSO algorithm
class PSO:

    def __init__(self, func, w, c1, c2, r1, r2, n):
        self.func = func
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.r1 = r1
        self.r2 = r2
        self.n = n
        self.Pbest = np.zeros(n)
        self.Gbest = 0.0
        self.Gfitness = 0.0

        # Arrays initialization
        ##########################################################
        # Creates the fisrt two random arrays with values: [0, 1)
        self.X1 = np.random.random(self.n)
        self.V1 = np.random.random(self.n)

        self.X2 = self.X1 * 10
        self.V2 = self.V1 * 10 

        # This are the ones that will be used
        self.X = self.X2 - 0.5
        self.V = self.V2 - 0.5
        ##########################################################


    # Calculates the next y
    def _vNext(self, x, v, Pbest):
        return ( self.w * v ) + ( self.c1 * self.r1 * ( Pbest - x ) ) + ( self.c2 * self.r2 * ( self.Gbest - x ) )  


    # Calculates the next x
    def _xNext(self, x, v):
        return x + v


    # Evaluates Pbest with X_fitness
    def _evaluate_Pbest(self):

        control = 0

        # Runs by all particles position
        for i in range(self.n):
            # If the current position is better than the older Pbest
            if self.func(self.X[i]) > self.func(self.Pbest[i]):
                self.Pbest[i] = self.X[i]
                control += 1

        # If some change was made
        if control > 0:
            return True
        else:
            return False


    # Function that executes pso algorithm for maximum of a function
    def pso(self):

        # Will controll the while loop
        Pbest_count = 0
        Pbest_cmp = self.Pbest
        cond = True

        # Fitness array of the position
        X_fitness = np.zeros(self.n)

        # Sets the first Pbest as the initial position
        self.Pbest = np.copy(self.X)

        # Calculates the fintness for the position
        for i in range(self.n):
            X_fitness[i] = self.func(self.X[i]) 

        # Sets the first Gbest as the position of the highest fitness    
        pos = np.where(X_fitness == np.max(X_fitness))
        self.Gbest = self.X[ pos[0][0] ]
        self.Gfitness = np.max(X_fitness)

        # Iteration loop (verificar se esta certo a condiçao)
        iteration = 0
        while cond or iteration < 10000:
            # For each particle
            for i in range(self.n):
                # Calculate the next position
                # v(i + 1) = w*v(i) + c1*r1*(Pbest - x(i)) + c2*r2*(Gbest - x(i))
                self.V[i] = self._vNext(self.X[i], self.V[i], self.Pbest[i])
                # x(i + 1) = x(i) + v(i + 1)
                self.X[i] = self._xNext(self.X[i], self.V[i])

                # Already calculate the new fitness state
                X_fitness[i] = self.func(self.X[i])

            # If neddes, update Pbest
            if self._evaluate_Pbest():
                # Verifies id Gbest must be updated
                val = np.max(X_fitness)
                if val > self.Gfitness:
                    pos = np.where(X_fitness == np.max(X_fitness))
                    self.Gbest = self.X[ pos[0][0] ]
                    self.Gfitness = val

                Pbest_count = 0

            # If Pbest did not change
            else:
                Pbest_count += 1
            
            # If the Pbest is't changed in 15 iterations, we assume that have converged
            if Pbest_count > 15:
                cond = False

            # Next iteration
            iteration += 1

            self.w -= 0.0001

        print("..:: The Maximum of the given function is: {:.4f} ::..".format(self.Gbest))
    

# Test bench
####################################################################
def func(x):
    return 1 + (27 * x) - (x ** 4)


def main(): 

    problem = PSO( func, 0.70 ,0.20 ,0.60 ,0.4657 ,0.5319 ,5 )

    problem.pso()


if __name__ == '__main__':
    main()
####################################################################