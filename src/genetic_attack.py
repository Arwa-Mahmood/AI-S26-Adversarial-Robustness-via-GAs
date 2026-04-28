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
        return np.random.uniform (-self.epsilon, self.epsilon, (self.pop_size, 784)).astype(np.float32)
        #astype fro converting datatypes -- faster + consistent 
        #ONE ROW -> ONE CANDIDATE 
    
    #------------------------------------------------------------------------------------------------------
    #FITNESS -- HOW MUCH DID WE CONFUSE THE MODEL. high = lower confidence = GOOD attack and vice versa
    #------------------------------------------------------------------------------------------------------

    def _fitness (self, population, x, true_label, predict_proba_fn):
        #apply each perturbation to x, clip to pixel range 
        adv_batch = np.clip(x + population, 0, 1)       #pop_size, 784 -- applying noise, keep pixels valid 
        probas = predict_proba_fn(adv_batch)            #pop_size, 10 -- MODEL CONFIDENCE FOR EACH CLASS
        #fitness = how much confidence lost on the true class 
        #higher = better for attacker 
        return 1.0 - probas[:, true_label]

    #------------------------------------------------------------------------------------------------------
    #TOURNAMENT --SELECTION
    #------------------------------------------------------------------------------------------------------

    def _selection (self, population, fitness, k = 3): 
        # Tourname Selection (K = )
        selected = []
        for _ in range (self.pop_size):
            #pick k random individuals, take the fittest
            idx = np.random.choice (self.pop_size, k, replace = False)  #random trio 
            winner = idx[np.argmax(fitness[idx])]       #picking indx w highest fitness -- fittest in the trio 
            selected.append(population[winner].copy()) 
        return np.array(selected)
    
    #------------------------------------------------------------------------------------------------------
    #UNIFORM --CROSSOVER
    # each of 784 genes independently inherited 
    #------------------------------------------------------------------------------------------------------
    def _crossover(self, parent1, parent2):
        # each gene inherited from either parent1 or parent2 w 50% probability 
        #Random bool mask 
        mask = np.random.rand (784) < 0.5       #50 percent true
        #if mask = true, take from parent 1 || ELSE -> PARENT2
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)       #complement -- children are mirrors 
        return child1, child2 
    
    #------------------------------------------------------------------------------------------------------
    #MUTATION -gaussian noise or random subset of genes 
    #------------------------------------------------------------------------------------------------------

    def _mutate(self, individual):
        mutated = individual.copy()

        #randomly picking genes to mutate -- 10 percent of pixels selected 
        mask = np.random.rand(784) < self.mutation_rate

        #add gaussian noise
        mutated[mask] += np.random.normal(0, self.epsilon*0.3, mask.sum())

        # ensures perturbation stays within allowed bounds 
        return np.clip(mutated, -self.epsilon, self.epsilon)
    

    #------------------------------------------------------------------------------------------------------    
    #SINGLE SAMPLE ATTACK 
    #------------------------------------------------------------------------------------------------------
    def attack (self, x, true_label, predict_fn, predict_proba_fn):
        """
        Attacks a single sample x.
        x -> INPUT IMAGE (flatened -- 784 vectors)
        true_label -- classes 0-9 
        predict_fn ->function that outputs class lebels 

        
        #Returns : (adversarial example, success_bool, generation_to_success)
        Returns: 
            adv_example : 784 adversarial image
            success     : bool -- was the classifier fooled? 
            gen_found   : int -- which generation it was fooled at (-1 if failed)
        """
        population = self._init_population()

        #evolve our generations
        for gen in range (self.n_generations):

            # evaluate fitness for every candidate in the population 
            fitness = self._fitness(population, x, true_label, predict_proba_fn)

            # early stop : check if best individual already fools the classifier 
            if self.early_stop:
                best_idx = np.argmax(fitness)
                adv = np.clip(x + population[best_idx], 0, 1)
                pred = predict_fn(adv.reshape(1,-1))[0] 

                #checking if attack succeeded
                if pred != true_label:
                    return adv, True, gen       #success -- exit early 
            
            # RUN TOURNAMENT SELECTION BEFORE CROSSOVER 
            population = self._selection(population, fitness)

            #crossover + mutation to build next generation  
            next_pop = []
            for i in range(0, self.pop_size -1, 2):
                p1, p2 = population[i], population[i+1]
                if np.random.rand() < self.crossover_rate: 
                    c1, c2 = self._crossover(p1, p2)
                else: 
                    c1, c2 = p1.copy(), p2.copy()       #skip crossover, keep parents 
                
                next_pop.append(self._mutate(c1))
                next_pop.append(self._mutate(c2))

            #keep population size exact (odd pop_size edge cases)
            next_pop = next_pop[:self.pop_size]         #trim in case pop_size is off 
            population = np.array(next_pop)

        # all generations exhasuted -- return best we found 
        fitness = self._fitness(population, x, true_label, predict_proba_fn)
        best_idx = np.argmax(fitness)
        adv = np.clip(x + population[best_idx],0 ,1)

        return adv, False, -1

        #Early Exit + Evolve 

    #------------------------------------------------------------------------------------------------------ 
    # DATASET ATTACK    
    #------------------------------------------------------------------------------------------------------    

    def attack_dataset(self, X, y, predict_fn, predict_proba_fn, 
                       n_samples=200, save_path = None, verbose = True):
        """
        Attack n_samples examples. Returns:
          - adversarial examples array
          - success flags
          - per-class attack success rates
        """

        adv_examples   = np.zeros((n_samples, 784), dtype = np.float32)
        success_flags  = np.zeros(n_samples, dtype = bool)
        gen_found      = np.full(n_samples, -1, dtype = int)

        for i in range(n_samples):
            adv, success, gen = self.attack(
                X[i], y[i], predict_fn, predict_proba_fn
            )
            adv_examples[i] = adv
            success_flags[i] = success 
            gen_found[i] = gen 

            if verbose and (i % 20 == 0 or i == n_samples -1): 
                asr = success_flags[: i+1].mean() * 100
                print(f"[{i+1}/{n_samples}] ASR so far: {asr:.1f}%")

        if save_path: 
            os.makedirs(os.path.dirname(save_path), exist_ok = True)
            np.savez(save_path, 
                     adv_examples = adv_examples, 
                     labels = y[: n_samples],
                     success_flags = success_flags,
                    gen_found = gen_found)
            
            print(f"[GA] Saved adversarial exmaples to {save_path}")

        return adv_examples, y[:n_samples], success_flags, gen_found

#------------------------------------------------------------------------------------------------------    
# EPSILON SWEEP UTILITY -- standalone function, not a method 
#------------------------------------------------------------------------------------------------------        
def epsilon_sweep (X, y, predict_fn, predict_proba_fn, 
                epsilons = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
                n_samples = 200, save_dir = 'results/'):
    
    os.makedirs(save_dir, exist_ok = True)
    results = {}

    #clean accuracy baseline 
    clean_preds = predict_fn(X[:n_samples])
    clean_acc = np.mean(clean_preds == y[:n_samples])
    print(f"[Sweep] Clean accuracy = {clean_acc: .4f}")

    for eps in epsilons: 
        print(f"\n[Sweep] epsilon = {eps}")
        ga = GeneticAttack (epsilon = eps, pop_size = 50, n_generations = 100)

        save_path = os.path.join(save_dir, f'adc_eps{eps}.npz')
        adv_X, labels, successes, gens = ga.attack_dataset(
            X, y, predict_fn, predict_proba_fn, 
            n_samples= n_samples, save_path = save_path
        )
            
        adv_preds = predict_fn(adv_X)
        adv_acc   = np.mean(adv_preds == labels)
        asr       = successes.mean() 

        results[eps] = {'asr' : asr, 'adv_acc' : adv_acc, 'clean_acc': clean_acc}
        print(f" ASR: {asr: .4f} | Adv Accuracy: {adv_acc: .4f}")

    return results 
    

#PIPELINE TEST 
if __name__ == "__main__":
    
    from decision_tree import DecisionTree 

   
    X_train = np.load('x_train_final_data.npy')
    X_test  = np.load('x_test_final_data.npy')
    y_train = np.load('y_train_final_data.npy')
    y_test  = np.load('y_test_final_data.npy')
    
    dt = DecisionTree.load ('models/decision_tree.pk1')     #load pretrained 

    #test single attack 
    ga = GeneticAttack (epsilon = 0.3, pop_size = 50, n_generations = 100)
    adv, success, gen = ga.attack (
        X_test[0], y_test[0], dt.predict, dt.predict_proba
        )
    
    print(f"Single attack | success: {success} | gen: {gen}")
    print(f"Original pred: {dt.predict(X_test[0:1])[0]} | Adv pred: {dt.predict(adv.reshape(1,-1))[0]}")

    # small sweep to verify pipeline
    results = epsilon_sweep(X_test, y_test, dt.predict, dt.predict_proba,
                            epsilons=[0.1, 0.3], n_samples=20)    
        
