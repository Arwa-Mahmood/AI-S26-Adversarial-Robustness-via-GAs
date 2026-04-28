"""
setup.py — run automatically by gui_temp.py on startup.
Checks for data + trained models. Trains anything that's missing.
Import this at the top of gui_temp.py: `import setup`
"""

import os
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))   # always relative to src/
os.makedirs("models", exist_ok=True)

# ─────────────────────────────────────────
# STEP 1: MNIST DATA
# ─────────────────────────────────────────
DATA_FILES = [
    'x_train_final_data.npy',
    'y_train_final_data.npy',
    'x_test_final_data.npy',
    'y_test_final_data.npy',
]

if not all(os.path.exists(f) for f in DATA_FILES):
    print("[setup] Data files not found. Extracting MNIST...")

    import os
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    from tensorflow.keras.datasets import mnist

    (x_train_raw, y_train_raw), _ = mnist.load_data()

    x_flat = x_train_raw.reshape(60000, 784) / 255.0

    selected_images, selected_labels = [], []
    for cls in range(10):
        idx = np.where(y_train_raw == cls)[0][:500]
        selected_images.append(x_flat[idx])
        selected_labels.append(y_train_raw[idx])

    x_data = np.concatenate(selected_images)
    y_data = np.concatenate(selected_labels)

    perm = np.random.permutation(len(x_data))
    x_data, y_data = x_data[perm], y_data[perm]

    np.save('x_train_final_data.npy', x_data[:4000])
    np.save('y_train_final_data.npy', y_data[:4000])
    np.save('x_test_final_data.npy',  x_data[4000:])
    np.save('y_test_final_data.npy',  y_data[4000:])

    print("[setup] MNIST data saved.")
else:
    print("[setup] Data files found.")

# ─────────────────────────────────────────
# STEP 2: DECISION TREE
# ─────────────────────────────────────────
if not os.path.exists('models/decision_tree.pkl'):
    print("[setup] Training Decision Tree...")

    from decision_tree import DecisionTree

    X_train = np.load('x_train_final_data.npy')
    y_train = np.load('y_train_final_data.npy')
    X_test  = np.load('x_test_final_data.npy')
    y_test  = np.load('y_test_final_data.npy')

    dt = DecisionTree(max_depth=20, min_samples_split=10, n_features=28)
    dt.fit(X_train, y_train)

    preds = dt.predict(X_test)
    print(f"[setup] DT Accuracy: {np.mean(preds == y_test):.4f}")

    dt.save('models/decision_tree.pkl')
    print("[setup] Decision Tree saved.")
else:
    print("[setup] Decision Tree found.")

# ─────────────────────────────────────────
# STEP 3: NEURAL NETWORK
# ─────────────────────────────────────────
if not os.path.exists('models/nn_model.keras'):
    print("[setup] Training Neural Network...")

    import os
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.utils import to_categorical

    X_train = np.load('x_train_final_data.npy')
    y_train = np.load('y_train_final_data.npy')
    X_test  = np.load('x_test_final_data.npy')
    y_test  = np.load('y_test_final_data.npy')

    y_train_cat = to_categorical(y_train, 10)
    y_test_cat  = to_categorical(y_test,  10)

    model = Sequential([
        Input(shape=(784,)),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(10,  activation='softmax'),
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, y_train_cat, epochs=10, batch_size=32, verbose=1)

    loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"[setup] NN Accuracy: {acc:.4f}")

    model.save('models/nn_model.keras')
    print("[setup] Neural Network saved.")
else:
    print("[setup] Neural Network found.")

print("[setup] All models ready.")