import numpy as np 

class GeneticAttack: 
    def __init__ (self, epsilon, pop_size = 50, n_generations = 100, 
                  mutation_rate = 0.1, crossover_rate = 0.7 ):
        #intialize 

    def _init_population (self) :
        #random perturbations in [-epsilon, epsilon]
        return np.random.uniform (-self.epsilon, self.epsilon, (self.popsize, 784))
    
    def _fitness (self, population, x, true_label, predict_proba_fn):
        # For Each Individual: 
        #   adv  = clip (x + delta, 0, 1)
        #   proba = predict_proba_fn(adv)
        #   fitness = 1 - proba [true_label] <- maximize confidence loss on true class 

    def _selection (self, population, fitness): 
        # Tourname Selection (K = )


    def _crossover (self, prant1, parent2): 
        #UNIFORM CROSSOVER: each gene from either p1 or p2 with x% probability 

    def _mutate (self, indvidual): 
        # Gaussian mutation on random subset of genes 

    def attack (self, x, true_label, predict_fn, predict_proba_fn):
        #Returns : (adversarial example, success_bool, generation_to_success)
    
        #Early Exit + Evolve 

    def attack_dataset(self, X, y, predict_fn, predict_proba_fn, n_samples=500):
        """
        Attack n_samples examples. Returns:
          - adversarial examples array
          - success flags
          - per-class attack success rates
        """
    
    
