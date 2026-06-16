# Adversarial Robustness — MNIST Attack & Defense
A study of how Genetic Algorithm based adversarial attacks can fool machine learning classifiers, and how well a Gaussian defense can recover them.
We trained two classifiers from scratch on handwritten digits, attacked them using a black-box Genetic Algorithm that evolves imperceptible noise, and tested a Gaussian smoothing defense to see which model could recover and why.

First run setup.py to ensure the model is trained
Then run the GUI file: load a digit, run the attack, apply the defense, and watch confidence scores update live.

## Project Files

| File | Purpose |
|------|-------------|
| `mnist.py` | Loads MNIST, normalises to [0,1], balances classes, saves train/test `.npy` arrays |
| `setup.py` | Checks for saved data and models — trains anything missing automatically |
| `neural_net.py` | Builds and trains the Keras NN |
| `genetic_attack.py` | The GA attack engine |
| `defense2.py` | Four defense strategies |
| `evaluate.py` | CLI pipeline covering all evaluation steps |
| `gui_temp.py` | Tkinter GUI with live confidence bars and animated NN diagram |

## How to run

**1. Install dependencies**
\```
pip install numpy tensorflow scikit-learn scipy pillow
\```

**2. Launch the demo**
\```
python gui.py
\```

that's it. setup.py runs automatically on launch and handles data extraction and model training before the GUI opens.
*Ayesha · Arwa · Eshaal*
