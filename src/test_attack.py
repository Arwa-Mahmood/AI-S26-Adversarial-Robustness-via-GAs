import numpy as np
from genetic_attack import GeneticAttack

# Test 1: Check population shape
ga = GeneticAttack()
pop = ga._init_population()
print(f"Population shape: {pop.shape}")  # Should be (100, 784)
print(f"Type of first element: {type(pop[0])}")  # Should be <class 'numpy.ndarray'>

# Test 2: Check crossover and mutation don't create nested lists
p1 = np.random.rand(784)
p2 = np.random.rand(784)
c1, c2 = ga._crossover(p1, p2)
print(f"Crossover outputs shapes: {c1.shape}, {c2.shape}")  # Both should be (784,)

# Test 3: Check attack runs without errors
from keras.models import load_model
X_test = np.load('x_test_final_data.npy')
y_test = np.load('y_test_final_data.npy')
nn_model = load_model('nn_model.keras')

def predict_fn(x):
    return np.argmax(nn_model.predict(x, verbose=0), axis=1)

def predict_proba_fn(x):
    return nn_model.predict(x, verbose=0)

# Test attack
adv, success, gen = ga.attack(X_test[0], y_test[0], predict_fn, predict_proba_fn)
print(f"Attack success: {success}")
print(f"Adversarial shape: {adv.shape}")  # Should be (784,)