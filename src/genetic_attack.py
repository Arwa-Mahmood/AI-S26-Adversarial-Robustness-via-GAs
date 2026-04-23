import numpy as np 
import os

class GeneticAttack: 
    def __init__ (self, epsilon = 0.2 , pop_size = 50, n_generations = 100, 
                  mutation_rate = 0.1, crossover_rate = 0.7, early_stop = True):
        self.epsilon = epsilon                      #max noise allowed   
        self.pop_size = pop_size                    #no. of candidate solutions per generation     
        self.n_generations = n_generations          #how long evaluations runs
        self. mutation_rate = mutation_rate         #random change 
        self.crossover_rate = crossover_rate        #probability of mixing parents        
        self.early_stop = early_stop                #stop if attack is successful early


    # POPULATION INIT
    def _init_population (self) :
        #random perturbations in [-epsilon, epsilon]
        return np.random.uniform (-self.epsilon, self.epsilon, (self.popsize, 784)).astype(np.float32)
        #astype fro converting datatypes -- faster + consistent 
        #ONE ROW -> ONE CANDIDATE 
    
    #FITNESS -- HOW MUCH DID WE CONFUSE THE MODEL. lower confidence = GOOD attack and vice versa
    def _fitness (self, population, x, true_label, predict_proba_fn):
        #apply each perturbation to x, clip to pixel range 
        adv_batch = np.clip(x + population, 0, 1)       #pop_size, 784
        probas = predict_proba_fn(adv_batch)            #pop_size, 10 -- MODEL CONFIDENCE FOR EACH CLASS
        #fitness = how much confidence lost on the true class 
        #higher = better for attacker 
        return 1.0 - probas[:, true_label]

    #TOURNAMENT --SELECTION
    def _selection (self, population, fitness, k = 3): 
        # Tourname Selection (K = )
        selected = []
        for _ in range (self.pop_size)
            #pick k random individuals, take the fittest
            idx = np.random.choice (self.pop_size, k, replace = False)
            winner = idx[np.argmax(fitness[idx])]   #picking indx w highest fitness
            selected.append(population[winner].copy()) 
        return np.array(selected)
    
    #UNIFORM --CROSSOVER
    def _crossover(self, parent1, parent2):
        # each gene inherited from either parent1 or parent2 w 50% probability 
        #Random bool mask 
        mask = np.random.rand (784) < 0.5       #50 percent true
        #if mask = true, take from parent 1 || ELSE -> PARENT2
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        return child1, child2 
    
    #MUTATION -gaussian noise or random subset of genes 
    def _mutate(self, individual):
        mutated = individual.copy()

        #randomly picking genes to mutate 
        mask = np.random.rand(784) < self.mutation_rate

        #add gaussian noise
        mutated[mask] += np.random.normal(0, self.epsilon*0.3, mask.sum())

        # ensures perturbation stays within allowed bounds 
        return np.clip(mutated, -self.epsilon, self.epsilon)
    
    #SINGLE SAMPLE ATTACK 
    def attack (self, x, true_label, predict_fn, predict_proba_fn):
        """
        Attacks a single sample x.

        #Returns : (adversarial example, success_bool, generation_to_success)
        Returns: 
            adv_example : 784 adversarial image
            success     : bool -- was the classifier fooled? 
            gen_found   : int -- which generation it was fooled at (-1 if failed)
        """
        population = self._init_population()

        #evolve our generations
        for gen in range (self.n_generations):

            #evaluate fitness
            fitness = self._fitness(population, x, true_label, predict_proba_fn)

            # early stop : check if best individual already fools the classifier 
            if self.early_stop:
                best_idx = np.argmax(fitness)
                adv = np.clip(x + population[best_idx], 0, 1)
                pred = predict_fn(adv.reshape(1,-1))[0] 
                if pred != true_label:
                    return adv, True, gen 

            #selection 
            next_pop = []
            for i in range(0, self.pop_size -1, 2):


        #Early Exit + Evolve 

    def attack_dataset(self, X, y, predict_fn, predict_proba_fn, n_samples=500):
        """
        Attack n_samples examples. Returns:
          - adversarial examples array
          - success flags
          - per-class attack success rates
        """
    
    
