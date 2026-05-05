import numpy as np
from keras.models import load_model
from genetic_attack import GeneticAttack
from defense import defend

# Load data and model
X_test = np.load('x_test_final_data.npy')
y_test = np.load('y_test_final_data.npy')
nn_model = load_model('nn_model.keras')

def predict_fn(x):
    return np.argmax(nn_model.predict(x, verbose=0), axis=1)

def predict_proba_fn(x):
    return nn_model.predict(x, verbose=0)

# Test on a few samples
n_test = 50
successful_attacks = 0
recovered = {sigma: 0 for sigma in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]}

ga = GeneticAttack(epsilon=0.3, pop_size=100, n_generations=150)

for i in range(n_test):
    print(f"Testing sample {i+1}/{n_test}")
    
    # Attack
    adv, success, _ = ga.attack(X_test[i], y_test[i], predict_fn, predict_proba_fn)
    
    if success:
        successful_attacks += 1
        
        # Test different defense strengths
        for sigma in recovered.keys():
            defended = defend(adv.reshape(1, -1), size=3, sigma=sigma)
            pred = predict_fn(defended)[0]
            if pred == y_test[i]:
                recovered[sigma] += 1

print(f"\nSuccessful attacks: {successful_attacks}/{n_test} ({successful_attacks/n_test*100:.1f}%)")
print("\nDefense Recovery Rates:")
for sigma, rec in recovered.items():
    if successful_attacks > 0:
        rate = rec / successful_attacks * 100
        print(f"Sigma={sigma}: {rec}/{successful_attacks} ({rate:.1f}%) recovered")