# Ant Colony Optimization (ACO)
# Luiz Felipe Raveduti Zafiro - 120513
# Artifitial Intelligence - Computer Engineering 
# Federal University of São Paulo (SJC) - 2020

import numpy as np

# Class that defines the ACO problem
class ACO:

    # Contructor
    def __init__(self, alpha, beta, rho, test=False):
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        # It will define if a given example will be executed
        self.test = test
        self.paths = []
        self.feromone = []
        self.size = 0
        self.visited = []
        self.prob = []


    # Generates a example
    def _generateExample(self):

        self.paths = [  [0.0,2.0,10.0,8.0,3.0],
                        [1.0,0.0,2.0,5.0,7.0],
                        [9.0,1.0,0.0,3.0,6.0],
                        [10.0,4.0,3.0,0.0,2.0],
                        [2.0,7.0,5.0,1.0,0.0]]

        self.feromone = [   [0.0,2.0,2.0,2.0,2.0],
                            [2.0,0.0,2.0,2.0,2.0],
                            [2.0,2.0,0.0,2.0,2.0],
                            [2.0,2.0,2.0,0.0,2.0],
                            [2.0,2.0,2.0,2.0,0.0]]

        self.paths = np.array(self.paths)
        self.feromone = np.array(self.feromone)
        self.size = 5


    # Calculates the probability of the i ant choose a path
    def _probCalc(self, i):

        denominator = 0.0

        self.prob = np.zeros(self.size, dtype=float)

        # Calculates the denominator
        for k in range(self.size):
            if k not in self.visited:
                denominator += ( self.alpha * self.feromone[i,k] ) * ( self.beta * self.paths[i,k] )

        for k in range(self.size):
            # 0 prob in a visited node
            if k in self.visited:
                self.prob[k] = 0.0

            else:
                numerator = ( self.alpha * self.feromone[i,k] ) * ( self.beta * self.paths[i,k] )
                self.prob[k] = numerator / denominator


    # Updates feromone for ant i
    def _feromoneUpdate(self, i):
        
        for j in range(self.size):
            # A path for it self doesnt exist
            if j != i:
                self.feromone[i,j] = ( (1 - self.rho) * self.feromone[i,j] ) + ( self.rho * ( 1 / self.paths[i,j] ) ) 

    
    # Choose next on the higest probability
    def _nextMove(self, prob):
        pos = np.where(prob == np.max(prob))
        return pos[0][0]
        

    def aco(self, startNode):

        # Stores current best path
        bestPath = []
        best_cont = 0

        # Runs the default example
        if self.test == True:
            self._generateExample()

        # Expect a input from user
        #else:
            #self._inputPaths()

        # Main loop -> until converge
        iter = 0
        # Normaly we use best_cont < 15, to check until converges
        # We are using iter < 1 just to evaluate the first iteration
        # Just as the example
        while iter < 1:
            # Iterates for all ants (nodes)
            for i in range(self.size):

                cont = 0
                j = i
                # Local current path
                curPath = []

                # caclulate and find the path
                while cont < self.size:
                    # To store current path, just append j
                    curPath.append(j)
                    # Mark i as visited
                    self.visited.append(j)
                    # Calculates the probability for all paths by ant i (return a array)
                    self._probCalc(j)
                    # New node of the path
                    j = self._nextMove(self.prob)
                    cont += 1

                # Update feromones for i'th ant
                self._feromoneUpdate(i)

                # If is the fist path calculated, it is the best
                if iter == 0 and i == startNode:
                    bestPath = curPath.copy()

                # Evaluates paths from the starting node
                if i == startNode:
                    if bestPath != curPath:
                        bestPath = curPath.copy()
                        best_cont = 0

                    else:
                        best_cont += 1

                # Frees visited list, for next ant
                self.visited = []

            iter += 1

        return bestPath


# Test Bench
#########################################################
def main():

    a = ACO(1, 1, 0.5, test=True)

    print(a.aco(0))
    print(a.paths)
    print(a.feromone)

if __name__ == '__main__':
    main()
#########################################################