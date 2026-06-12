"""
  1. Baseline accuracy  (DT + NN on clean test set)
  2. GA attack          (DT + NN, configurable epsilon / n_samples)
  3. Gaussian defense   (multiple sigma values)
  4. Summary table printed to stdout + saved as results/summary.txt
"""

import os, argparse
import numpy as np
import pickle
import sys


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

from decision_tree import DecisionTree

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("results", exist_ok=True)


# CLI ARGS
parser = argparse.ArgumentParser()
parser.add_argument("--eps",    type=float, default=0.3,  
                    help="GA epsilon (perturbation budget)")
parser.add_argument("--n",      type=int,   default=100,
                    help="Number of test samples to attack")
parser.add_argument("--pop",    type=int,   default=100,  
                    help="GA population size")
parser.add_argument("--gens",   type=int,   default=150,  
                    help="GA generations")
parser.add_argument("--sigmas", type=float, nargs="+",
                    default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],  
                    help="Sigma values for Gaussian defense sweep")
args = parser.parse_args()



# Loading Data 
print("[eval] Loading data...")
X_test  = np.load("x_test_final_data.npy")
y_test  = np.load("y_test_final_data.npy")

N = min(args.n, len(X_test))
X_eval = X_test[:N]
y_eval = y_test[:N]

# Loading models 
print("[eval] Loading models...")

import decision_tree
sys.modules['__main__'].Node = Node  # This helps pickle find Node class

with open("models/decision_tree.pkl", "rb") as f:
    dt_model = pickle.load(f)

def dt_predict(X):
    """Batch label prediction."""
    return dt_model.predict(X)

def dt_predict_proba(X):
    """Batch probability prediction — shape (N, 10)."""
    return dt_model.predict_proba(X)

# Neural Network
import os; os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from tensorflow.keras.models import load_model as keras_load

nn_model = keras_load("models/nn_model.keras")

def nn_predict(X):
    return np.argmax(nn_model.predict(X, verbose=0), axis=1)

def nn_predict_proba(X):
    return nn_model.predict(X, verbose=0)

# Defense
from defense2 import EnhancedDefense

# Genetic Attack
from genetic_attack import GeneticAttack


# Helper Functions 

def accuracy(preds, labels):
    return float(np.mean(preds == labels)) * 100

def run_ga_attack(X, y, predict_fn, predict_proba_fn, label=""):
    
    #Runs GA attack on X[i] for i in range(len(X)).
    #Returns adversarial examples array and success flags.
    
    ga = GeneticAttack(
        epsilon=args.eps,
        pop_size=args.pop,
        n_generations=args.gens,
        mutation_rate=0.15,
        crossover_rate=0.8,
        early_stop=True,
    )

    adv_examples  = np.zeros_like(X)
    success_flags = np.zeros(len(X), dtype=bool)

    for i in range(len(X)):
        adv, success, gen = ga.attack(X[i], y[i], predict_fn, predict_proba_fn)
        adv_examples[i]  = adv
        success_flags[i] = success

        if (i + 1) % 20 == 0 or i == len(X) - 1:
            asr = success_flags[: i + 1].mean() * 100
            print(f"  [{label}] [{i+1}/{len(X)}] ASR so far: {asr:.1f}%")

    return adv_examples, success_flags

def defense_sweep(X_adv, y, predict_fn, sigmas, label=""):

    defense = EnhancedDefense()
    rows = []
    
    for sigma in sigmas:
        X_smooth = defense.defend(X_adv, method='ensemble', sigma=sigma)
        preds = predict_fn(X_smooth)
        acc = accuracy(preds, y)
        rows.append((sigma, acc))
        print(f"  [{label}] sigma={sigma:.1f} → acc after defense: {acc:.2f}%")
    
    return rows


# 1. BASELINE ACCURACY
print("\n" + "="*55)
print("  STEP 1 — Baseline accuracy (clean test set)")
print("="*55)

dt_clean_preds = dt_predict(X_eval)
nn_clean_preds = nn_predict(X_eval)
dt_clean_acc   = accuracy(dt_clean_preds, y_eval)
nn_clean_acc   = accuracy(nn_clean_preds, y_eval)

print(f"  DT clean accuracy : {dt_clean_acc:.2f}%")
print(f"  NN clean accuracy : {nn_clean_acc:.2f}%")


# 2. GA ATTACK ON DECISION TREE
print("\n" + "="*55)
print(f"  STEP 2a — GA attack on DT  (ε={args.eps}, {N} samples)")
print("="*55)

adv_dt, flags_dt = run_ga_attack(
    X_eval, y_eval,
    predict_fn        = lambda X: dt_predict(X),
    predict_proba_fn  = lambda X: dt_predict_proba(X),
    label="DT"
)

dt_adv_preds_dt = dt_predict(adv_dt)
dt_adv_preds_nn = nn_predict(adv_dt)

dt_adv_acc_dt = accuracy(dt_adv_preds_dt, y_eval)
dt_adv_acc_nn = accuracy(dt_adv_preds_nn, y_eval)
dt_asr        = flags_dt.mean() * 100

print(f"\n  [DT attack] ASR on DT              : {dt_asr:.2f}%")
print(f"  [DT attack] DT acc after attack    : {dt_adv_acc_dt:.2f}%")
print(f"  [DT attack] NN acc on DT adversarials: {dt_adv_acc_nn:.2f}%")

np.savez("results/adv_dt.npz",
         adv=adv_dt, labels=y_eval, success=flags_dt)


# 3. GA ATTACK ON NEURAL NETWORK
print("\n" + "="*55)
print(f"  STEP 2b — GA attack on NN  (ε={args.eps}, {N} samples)")
print("="*55)

adv_nn, flags_nn = run_ga_attack(
    X_eval, y_eval,
    predict_fn        = lambda X: nn_predict(X),
    predict_proba_fn  = lambda X: nn_predict_proba(X),
    label="NN"
)

nn_adv_preds_nn = nn_predict(adv_nn)
nn_adv_preds_dt = dt_predict(adv_nn)

nn_adv_acc_nn = accuracy(nn_adv_preds_nn, y_eval)
nn_adv_acc_dt = accuracy(nn_adv_preds_dt, y_eval)
nn_asr        = flags_nn.mean() * 100

print(f"\n  [NN attack] ASR on NN              : {nn_asr:.2f}%")
print(f"  [NN attack] NN acc after attack    : {nn_adv_acc_nn:.2f}%")
print(f"  [NN attack] DT acc on NN adversarials: {nn_adv_acc_dt:.2f}%")

np.savez("results/adv_nn.npz",
         adv=adv_nn, labels=y_eval, success=flags_nn)


# 4. GAUSSIAN DEFENSE ON DT ADVERSARIALS
print("\n" + "="*55)
print("  STEP 3a — Gaussian defense on DT adversarials")
print("="*55)

defense_dt_on_dt = defense_sweep(adv_dt, y_eval, dt_predict, args.sigmas, "DT→DT")
defense_dt_on_nn = defense_sweep(adv_dt, y_eval, nn_predict, args.sigmas, "DT→NN")


# 5. GAUSSIAN DEFENSE ON NN ADVERSARIALS
print("\n" + "="*55)
print("  STEP 3b — Gaussian defense on NN adversarials")
print("="*55)

defense_nn_on_nn = defense_sweep(adv_nn, y_eval, nn_predict, args.sigmas, "NN→NN")
defense_nn_on_dt = defense_sweep(adv_nn, y_eval, dt_predict, args.sigmas, "NN→DT")


# 6. SUMMARY TABLE
lines = []
lines.append("=" * 65)
lines.append("  ADVERSARIAL ROBUSTNESS — EVALUATION SUMMARY")
lines.append(f"  ε={args.eps} | pop={args.pop} | gens={args.gens} | N={N}")
lines.append("=" * 65)

lines.append("\n  BASELINE ACCURACY (clean images)")
lines.append(f"    DT : {dt_clean_acc:.2f}%")
lines.append(f"    NN : {nn_clean_acc:.2f}%")

lines.append("\n  AFTER GA ATTACK")
lines.append(f"    {'Model':<8} {'ASR':>8} {'Acc after atk':>16}")
lines.append(f"    {'-'*35}")
lines.append(f"    {'DT':<8} {dt_asr:>7.2f}% {dt_adv_acc_dt:>15.2f}%")
lines.append(f"    {'NN':<8} {nn_asr:>7.2f}% {nn_adv_acc_nn:>15.2f}%")

lines.append("\n  TRANSFERABILITY (cross-model accuracy on adversarials)")
lines.append(f"    DT adversarials evaluated by NN : {dt_adv_acc_nn:.2f}%")
lines.append(f"    NN adversarials evaluated by DT : {nn_adv_acc_dt:.2f}%")

lines.append("\n  GAUSSIAN DEFENSE — DT adversarials → DT")
lines.append(f"    {'sigma':>8} {'acc':>10}")
for sigma, acc in defense_dt_on_dt:
    lines.append(f"    {sigma:>8.1f} {acc:>9.2f}%")

lines.append("\n  GAUSSIAN DEFENSE — DT adversarials → NN")
lines.append(f"    {'sigma':>8} {'acc':>10}")
for sigma, acc in defense_dt_on_nn:
    lines.append(f"    {sigma:>8.1f} {acc:>9.2f}%")

lines.append("\n  GAUSSIAN DEFENSE — NN adversarials → NN")
lines.append(f"    {'sigma':>8} {'acc':>10}")
for sigma, acc in defense_nn_on_nn:
    lines.append(f"    {sigma:>8.1f} {acc:>9.2f}%")

lines.append("\n  GAUSSIAN DEFENSE — NN adversarials → DT")
lines.append(f"    {'sigma':>8} {'acc':>10}")
for sigma, acc in defense_nn_on_dt:
    lines.append(f"    {sigma:>8.1f} {acc:>9.2f}%")

lines.append("\n" + "=" * 65)

summary = "\n".join(lines)
print("\n" + summary)

with open("results/summary.txt", "w", encoding="utf-8") as f:
    f.write(summary)

print("\n[eval] Done. Summary saved to results/summary.txt")