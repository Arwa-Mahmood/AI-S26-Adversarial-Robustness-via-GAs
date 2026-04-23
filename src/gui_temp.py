import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pickle
import time

# --- Load data ---
X_test = np.load('x_test_final_data.npy')
y_test = np.load('y_test_final_data.npy')

# --- Load decision tree if exists, else stub ---
try:
    with open('models/decision_tree.pkl', 'rb') as f:
        dt_model = pickle.load(f)
    def dt_predict(image):
        return dt_model.predict_proba(image.reshape(1, -1))[0]
    print("DT loaded.")
except:
    def dt_predict(image):
        return np.random.dirichlet(np.ones(10))
    print("DT not found, using stub.")

# --- Stubs — replace with real imports when ready ---
def nn_predict(image):
    return np.random.dirichlet(np.ones(10))

def attack(predict_fn, image, label, epsilon):
    noise = np.random.uniform(-epsilon, epsilon, image.shape)
    return np.clip(image + noise, 0, 1)

def apply_smoothing(image, sigma):
    return image

# --- State ---
current = {
    "image": None,
    "label": None,
    "adversarial": None,
    "defended": None
}

MNIST_CLASSES = [str(i) for i in range(10)]

# ─────────────────────────────────────────
# NEURAL NETWORK DIAGRAM
# ─────────────────────────────────────────
# Layer sizes — matches your actual NN architecture
# Input(784) -> Dense(256) -> Dense(128) -> Output(10)
# We display a compressed version: 6 input nodes, 6 hidden1, 5 hidden2, 4 output
LAYER_SIZES    = [6, 6, 5, 4]
LAYER_LABELS   = ["Input\n(784)", "Dense\n(256)", "Dense\n(128)", "Output\n(10)"]
LAYER_COLORS   = ["#4a90d9", "#e09a3a", "#9b59b6", "#4caf7d"]
NEURON_RADIUS  = 12
CANVAS_W       = 420
CANVAS_H       = 260
ANIM_DELAY     = 60   # ms between animation frames

def get_neuron_positions():
    """Returns dict: layer_idx -> list of (x, y) center positions"""
    positions = {}
    x_gap = CANVAS_W // (len(LAYER_SIZES) + 1)
    for li, size in enumerate(LAYER_SIZES):
        x = x_gap * (li + 1)
        y_gap = CANVAS_H // (size + 1)
        positions[li] = [(x, y_gap * (ni + 1)) for ni in range(size)]
    return positions

NEURON_POS = get_neuron_positions()

def draw_network(canvas, active_layer=-1, activations=None):
    """
    Draws the full network on the tk canvas.
    active_layer: which layer is currently lit up (-1 = none)
    activations: list of floats (0-1) per neuron in active layer
    """
    canvas.delete("all")

    # Draw connections first (behind neurons)
    for li in range(len(LAYER_SIZES) - 1):
        for (x1, y1) in NEURON_POS[li]:
            for (x2, y2) in NEURON_POS[li + 1]:
                # light up connections leading INTO active layer
                if active_layer == li + 1:
                    color = "#f0c040"
                    width = 1.5
                else:
                    color = "#cccccc"
                    width = 0.8
                canvas.create_line(x1, y1, x2, y2,
                                   fill=color, width=width)

    # Draw neurons
    for li, positions in NEURON_POS.items():
        for ni, (x, y) in enumerate(positions):
            r = NEURON_RADIUS

            # Pick fill color
            if li == active_layer:
                if activations is not None and ni < len(activations):
                    intensity = activations[ni]
                    # blend from gray to layer color based on intensity
                    fill = blend_color("#dddddd", LAYER_COLORS[li], intensity)
                else:
                    fill = LAYER_COLORS[li]
                outline = "white"
                width = 2
            else:
                fill = "#e8e8e8"
                outline = "#aaaaaa"
                width = 1

            canvas.create_oval(x - r, y - r, x + r, y + r,
                                fill=fill, outline=outline, width=width)

    # Draw layer labels at bottom
    x_gap = CANVAS_W // (len(LAYER_SIZES) + 1)
    for li, label in enumerate(LAYER_LABELS):
        x = x_gap * (li + 1)
        canvas.create_text(x, CANVAS_H - 8, text=label,
                           font=("Arial", 7), fill="#555555", justify=tk.CENTER)

def blend_color(hex1, hex2, t):
    """Blend two hex colors. t=0 -> hex1, t=1 -> hex2"""
    t = max(0.0, min(1.0, t))
    r1, g1, b1 = int(hex1[1:3],16), int(hex1[3:5],16), int(hex1[5:7],16)
    r2, g2, b2 = int(hex2[1:3],16), int(hex2[3:5],16), int(hex2[5:7],16)
    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)
    return f"#{r:02x}{g:02x}{b:02x}"

def animate_network(image):
    """
    Animates data flowing through the network layer by layer.
    Uses fake activations scaled from image stats — 
    swap with real layer activations when neural_net.py is ready.
    """
    # Generate fake activations per layer from image
    # Replace these with real intermediate layer outputs later
    fake_activations = [
        np.abs(np.random.randn(LAYER_SIZES[0])),   # input
        np.abs(np.random.randn(LAYER_SIZES[1])),   # dense 1
        np.abs(np.random.randn(LAYER_SIZES[2])),   # dense 2
        nn_predict(image)[:LAYER_SIZES[3]]          # output — real probs
    ]

    # Normalize each layer's activations to 0-1
    for i in range(len(fake_activations)):
        mx = fake_activations[i].max()
        if mx > 0:
            fake_activations[i] = fake_activations[i] / mx

    def step(layer_idx):
        if layer_idx >= len(LAYER_SIZES):
            # Animation done — leave output layer lit
            draw_network(nn_canvas, active_layer=len(LAYER_SIZES)-1,
                         activations=fake_activations[-1])
            return
        draw_network(nn_canvas, active_layer=layer_idx,
                     activations=fake_activations[layer_idx])
        window.after(speed_var.get(), lambda: step(layer_idx + 1))

    step(0)

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────

def show_image(canvas, image_array):
    img = image_array.reshape(28, 28)
    img = (img * 255).astype(np.uint8)
    pil_img = Image.fromarray(img, mode='L')
    pil_img = pil_img.resize((150, 150), Image.NEAREST)
    photo = ImageTk.PhotoImage(pil_img)
    canvas.configure(image=photo)
    canvas.image = photo

def update_predictions(image, dt_label, nn_label, true_label):
    dt_probs = dt_predict(image)
    dt_pred  = np.argmax(dt_probs)
    dt_conf  = int(dt_probs[dt_pred] * 100)
    dt_label.config(
        text=f"DT: {dt_pred} ({dt_conf}%)",
        fg="green" if dt_pred == true_label else "red"
    )

    nn_probs = nn_predict(image)
    nn_pred  = np.argmax(nn_probs)
    nn_conf  = int(nn_probs[nn_pred] * 100)
    nn_label.config(
        text=f"NN: {nn_pred} ({nn_conf}%)",
        fg="green" if nn_pred == true_label else "red"
    )
    window.update()
'''
def update_histogram():
    ax.clear()
    ax.set_facecolor("#f5f5f5")
    ax.set_xlabel("Pixel Intensity", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    ax.tick_params(labelsize=7)

    if current["image"] is not None:
        ax.hist(current["image"], bins=50, alpha=0.6,
                color="steelblue", label="Original")
    if current["adversarial"] is not None:
        ax.hist(current["adversarial"], bins=50, alpha=0.6,
                color="tomato", label="After Attack")
    if current["defended"] is not None:
        ax.hist(current["defended"], bins=50, alpha=0.6,
                color="mediumseagreen", label="After Defense")

    ax.legend(fontsize=7)
    fig.tight_layout()
    hist_canvas.draw()
'''
# ─────────────────────────────────────────
# BUTTON FUNCTIONS
# ─────────────────────────────────────────

def load_image():
    idx = np.random.randint(0, len(X_test))
    current["image"]       = X_test[idx]
    current["label"]       = int(y_test[idx])
    current["adversarial"] = None
    current["defended"]    = None

    show_image(original_canvas, current["image"])
    update_predictions(current["image"],
                       orig_dt_label, orig_nn_label, current["label"])
    true_label_display.config(text=f"True Label: {current['label']}")

    attack_canvas.configure(image='')
    defense_canvas.configure(image='')
    for lbl in [attack_dt_label, attack_nn_label,
                defense_dt_label, defense_nn_label]:
        lbl.config(text="—", fg="black")

    #update_histogram()
    animate_network(current["image"])
    status_label.config(text="Image loaded. Ready to attack.")

def run_attack():
    if current["image"] is None:
        status_label.config(text="Load an image first.")
        return
    status_label.config(text="Running attack... please wait.")
    window.update()

    adv = attack(dt_predict, current["image"],
                 current["label"], epsilon_var.get())
    current["adversarial"] = adv

    show_image(attack_canvas, adv)
    update_predictions(adv, attack_dt_label, attack_nn_label, current["label"])
    #update_histogram()
    animate_network(adv)
    status_label.config(text="Attack done. Red = fooled, Green = survived.")

def run_defense():
    if current["adversarial"] is None:
        status_label.config(text="Run attack first.")
        return
    status_label.config(text="Applying defense...")
    window.update()

    defended = apply_smoothing(current["adversarial"], sigma_var.get())
    current["defended"] = defended

    show_image(defense_canvas, defended)
    update_predictions(defended, defense_dt_label,
                       defense_nn_label, current["label"])
    #update_histogram()
    animate_network(defended)
    status_label.config(text="Defense done. Green = recovered, Red = still fooled.")

def reset():
    current.update({"image": None, "label": None,
                    "adversarial": None, "defended": None})
    for canvas in [original_canvas, attack_canvas, defense_canvas]:
        canvas.configure(image='')
    for lbl in [orig_dt_label, orig_nn_label,
                attack_dt_label, attack_nn_label,
                defense_dt_label, defense_nn_label]:
        lbl.config(text="—", fg="black")
    true_label_display.config(text="True Label: —")
    draw_network(nn_canvas)
    #ax.clear()
    #ax.set_title("Load an image to begin", fontsize=8)
    #hist_canvas.draw()
    status_label.config(text="Reset.")

# ─────────────────────────────────────────
# WINDOW
# ─────────────────────────────────────────

window = tk.Tk()
window.title("Adversarial Attack & Defense Demo")
window.geometry("1200x780")
window.configure(bg="#E3F6EC")

# ─────────────────────────────────────────
# LEFT PANEL — controls
# ─────────────────────────────────────────

left_panel = tk.Frame(window, bg="#E3F6EC", width=175)
left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=12, pady=12)

tk.Label(left_panel, text="Controls",
         font=("Arial", 12, "bold"), bg="#E3F6EC").pack(pady=8)

tk.Label(left_panel, text="Epsilon:",
         bg="#E3F6EC", font=("Arial", 9)).pack(pady=(12, 0))
epsilon_var = tk.DoubleVar(value=0.1)
tk.Scale(left_panel, from_=0.05, to=0.25, resolution=0.05,
         orient=tk.HORIZONTAL, variable=epsilon_var,
         bg="#E3F6EC", length=150).pack()

tk.Label(left_panel, text="Sigma:",
         bg="#E3F6EC", font=("Arial", 9)).pack(pady=(12, 0))
sigma_var = tk.DoubleVar(value=1.0)
tk.Scale(left_panel, from_=0.5, to=2.0, resolution=0.5,
         orient=tk.HORIZONTAL, variable=sigma_var,
         bg="#E3F6EC", length=150).pack()

true_label_display = tk.Label(left_panel, text="True Label: —",
                               bg="#E3F6EC", font=("Arial", 10, "bold"))
true_label_display.pack(pady=(18, 5))

tk.Button(left_panel, text="Load Random Image",
          command=load_image, width=17,
          bg="#4a90d9", fg="white", relief=tk.FLAT).pack(pady=(8, 4))
tk.Button(left_panel, text="Run Attack",
          command=run_attack, width=17,
          bg="#e05555", fg="white", relief=tk.FLAT).pack(pady=4)
tk.Button(left_panel, text="Apply Defense",
          command=run_defense, width=17,
          bg="#4caf7d", fg="white", relief=tk.FLAT).pack(pady=4)
tk.Button(left_panel, text="Reset",
          command=reset, width=17,
          bg="#aaaaaa", fg="white", relief=tk.FLAT).pack(pady=4)

# ─────────────────────────────────────────
# RIGHT AREA
# ─────────────────────────────────────────

right_area = tk.Frame(window, bg="#E3F6EC")
right_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=12, padx=8)

# ── TOP: images ──
image_panel = tk.Frame(right_area, bg="#E3F6EC")
image_panel.pack(fill=tk.X)

for col, title in enumerate(["Original", "After Attack", "After Defense"]):
    tk.Label(image_panel, text=title, bg="#E3F6EC",
             font=("Arial", 10, "bold")).grid(row=0, column=col, padx=20)

original_canvas = tk.Label(image_panel, bg="white", width=150, height=150)
original_canvas.grid(row=1, column=0, padx=20, pady=6)
attack_canvas   = tk.Label(image_panel, bg="white", width=150, height=150)
attack_canvas.grid(row=1, column=1, padx=20, pady=6)
defense_canvas  = tk.Label(image_panel, bg="white", width=150, height=150)
defense_canvas.grid(row=1, column=2, padx=20, pady=6)

orig_dt_label    = tk.Label(image_panel, text="DT: —", bg="#E3F6EC", font=("Arial", 9))
orig_dt_label.grid(row=2, column=0)
attack_dt_label  = tk.Label(image_panel, text="DT: —", bg="#E3F6EC", font=("Arial", 9))
attack_dt_label.grid(row=2, column=1)
defense_dt_label = tk.Label(image_panel, text="DT: —", bg="#E3F6EC", font=("Arial", 9))
defense_dt_label.grid(row=2, column=2)

orig_nn_label    = tk.Label(image_panel, text="NN: —", bg="#E3F6EC", font=("Arial", 9))
orig_nn_label.grid(row=3, column=0)
attack_nn_label  = tk.Label(image_panel, text="NN: —", bg="#E3F6EC", font=("Arial", 9))
attack_nn_label.grid(row=3, column=1)
defense_nn_label = tk.Label(image_panel, text="NN: —", bg="#E3F6EC", font=("Arial", 9))
defense_nn_label.grid(row=3, column=2)

# ── BOTTOM: NN diagram + histogram side by side ──
bottom_area = tk.Frame(right_area, bg="#E3F6EC")
bottom_area.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

# NN canvas on the left
nn_frame = tk.Frame(bottom_area, bg="#E3F6EC")
nn_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))

tk.Label(nn_frame, text="Neural Network",
         bg="#E3F6EC", font=("Arial", 9, "bold")).pack()
nn_canvas = tk.Canvas(nn_frame, width=CANVAS_W, height=CANVAS_H,
                      bg="#f8f8f8", highlightthickness=1,
                      highlightbackground="#cccccc")
nn_canvas.pack()

# Controls under the NN diagram
nn_controls = tk.Frame(nn_frame, bg="#E3F6EC")
nn_controls.pack(fill=tk.X, pady=(4, 0))

tk.Label(nn_controls, text="Speed:",
         bg="#E3F6EC", font=("Arial", 8)).pack(side=tk.LEFT, padx=(0, 4))
speed_var = tk.IntVar(value=60)
tk.Scale(nn_controls, from_=10, to=300, resolution=10,
         orient=tk.HORIZONTAL, variable=speed_var,
         bg="#E3F6EC", length=150,
         label="fast         slow").pack(side=tk.LEFT)

tk.Button(nn_controls, text="Replay",
          command=lambda: animate_network(current["image"]) if current["image"] is not None else None,
          bg="#4a90d9", fg="white", relief=tk.FLAT,
          font=("Arial", 8), width=8).pack(side=tk.LEFT, padx=(10, 0))

# Histogram on the right
'''
hist_frame = tk.Frame(bottom_area, bg="#E3F6EC")
hist_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

tk.Label(hist_frame, text="Pixel Intensity Distribution",
         bg="#E3F6EC", font=("Arial", 9, "bold")).pack()

fig, ax = plt.subplots(figsize=(5, 2.8))
fig.patch.set_facecolor("#E3F6EC")
ax.set_facecolor("#f5f5f5")
ax.set_title("Load an image to begin", fontsize=8)

hist_canvas = FigureCanvasTkAgg(fig, master=hist_frame)
hist_canvas.draw()
hist_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
'''
# ─────────────────────────────────────────
# STATUS BAR
# ─────────────────────────────────────────

status_label = tk.Label(window, text="Run mnist.py first, then launch this.",
                        bg="#dddddd", anchor="w", font=("Arial", 9))
status_label.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=3)

# ── Draw empty network on startup ──
draw_network(nn_canvas)

window.mainloop()
