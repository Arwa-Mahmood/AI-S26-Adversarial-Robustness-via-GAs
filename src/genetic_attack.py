import numpy as np 
import os

class GeneticAttack: 
    def __init__ (self, epsilon = 0.3 , pop_size = 100, n_generations = 200, 
                  mutation_rate = 0.15, crossover_rate = 0.7, early_stop = True):
        self.epsilon = epsilon                      #max noise allowed   
        self.pop_size = pop_size                    #no. of candidate solutions per generation     
        self.n_generations = n_generations          #how long evaluations runs
        self. mutation_rate = mutation_rate         #random change 
        self.crossover_rate = crossover_rate        #probability of mixing parents        
        self.early_stop = early_stop                #stop if attack is successful early


    def _init_population (self) :
        return np.random.uniform (-self.epsilon, self.epsilon, (self.pop_size, 784)).astype(np.float32)
    

    def _fitness (self, population, x, true_label, predict_proba_fn):
        #apply each perturbation to x, clip to pixel range 
        adv_batch = np.clip(x + population, 0, 1)        
        probas = predict_proba_fn(adv_batch)            
    
        true_class_confidence = probas[:, true_label]
        max_other_confidence = np.max(probas[:, [i for i in range(10) if i != true_label]], axis=1)
        fitness = (1.0 - true_class_confidence) + (max_other_confidence * 0.5)
        return fitness
        # low confidence on true label and high confidence on wrong label


    #tournament and selection 
    def _selection (self, population, fitness, k = 3): 
        selected = []
        for _ in range (self.pop_size):
            #pick k random individuals and take the fittest
            idx = np.random.choice (self.pop_size, k, replace = False)  
            winner = idx[np.argmax(fitness[idx])]        
            selected.append(population[winner].copy()) 
        return np.array(selected)
    
    
    #UNIFORM --CROSSOVER
    def _crossover(self, parent1, parent2):
        mask = np.random.rand (784) < 0.5      
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)       
        return child1, child2 
    
    
    #mutation -gaussian noise or random subset of genes 

    def _mutate(self, individual):
        mutated = individual.copy()

        #randomly picking genes to mutate -- 10 percent of pixels selected 
        mask = np.random.rand(784) < self.mutation_rate

        #add gaussian noise
        mutated[mask] += np.random.normal(0, self.epsilon*0.3, mask.sum())

        # ensures perturbation stays within allowed bounds 
        return np.clip(mutated, -self.epsilon, self.epsilon)
    

        
    def attack (self, x, true_label, predict_fn, predict_proba_fn):
        
        # attacks single cample -> flattened to vectors -> true_label -> classes 0-9 
        # returns adv_example, success, and gen_found 

        population = self._init_population()

        #evolving generations 
        for gen in range (self.n_generations):

            fitness = self._fitness(population, x, true_label, predict_proba_fn)

            # early stop : check if best individual already fools the classifier 
            if self.early_stop:
                best_idx = np.argmax(fitness)
                adv = np.clip(x + population[best_idx], 0, 1)
                pred = predict_fn(adv.reshape(1,-1))[0] 

                #checking if attack succeeded
                if pred != true_label:
                    return adv, True, gen     
            
            # Running tournament selection before crossover 
            population = self._selection(population, fitness)

            #crossover + mutation to build next generation  
            next_pop = []
            for i in range(0, self.pop_size -1, 2):
                p1, p2 = population[i], population[i+1]
                if np.random.rand() < self.crossover_rate: 
                    c1, c2 = self._crossover(p1, p2)
                else: 
                    c1, c2 = p1.copy(), p2.copy()       
                
                next_pop.append(self._mutate(c1))
                next_pop.append(self._mutate(c2))


            if len(next_pop) < self.pop_size:
                next_pop.append(self._mutate(population[-1]))

            population = np.array(next_pop[:self.pop_size])


        # all generations exhasuted -- return best we found 
        fitness = self._fitness(population, x, true_label, predict_proba_fn)
        best_idx = np.argmax(fitness)
        adv = np.clip(x + population[best_idx],0 ,1)

        for _ in range(5000):
            noise = np.random.uniform(-self.epsilon, self.epsilon, x.shape)
            candidate = np.clip(x + noise, 0, 1)
            if predict_fn(candidate.reshape(1, -1))[0] != true_label:
                return candidate, True, -2
            
        return adv, False, -1
        #Early Exit + Evolve 

     
    # Attacking dataset   
    def attack_dataset(self, X, y, predict_fn, predict_proba_fn, 
                       n_samples=200, save_path = None, verbose = True):
        
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
    


# pipeline test 
if __name__ == "__main__":
    
    from decision_tree import DecisionTree 
   
    X_train = np.load('x_train_final_data.npy')
    X_test  = np.load('x_test_final_data.npy')
    y_train = np.load('y_train_final_data.npy')
    y_test  = np.load('y_test_final_data.npy')
    
    dt = DecisionTree.load ('models/decision_tree.pkl') 

    #testing single attack 
    ga = GeneticAttack (epsilon = 0.3, pop_size = 50, n_generations = 100)
    adv, success, gen = ga.attack (
        X_test[0], y_test[0], dt.predict, dt.predict_proba
        )
    
    print(f"Single attack | success: {success} | gen: {gen}")
    print(f"Original pred: {dt.predict(X_test[0:1])[0]} | Adv pred: {dt.predict(adv.reshape(1,-1))[0]}")

    # small sweeping to verify pipeline
    results = epsilon_sweep(X_test, y_test, dt.predict, dt.predict_proba,
                            epsilons=[0.1, 0.3], n_samples=20)