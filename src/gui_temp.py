# import tkinter as tk
# from tkinter import ttk
# from PIL import Image, ImageTk
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# import pickle
# import time

# # --- Load data ---
# X_test = np.load('x_test_final_data.npy')
# y_test = np.load('y_test_final_data.npy')

# # --- Load decision tree if exists, else stub ---
# try:
#     with open('models/decision_tree.pkl', 'rb') as f:
#         dt_model = pickle.load(f)
#     def dt_predict(image):
#         return dt_model.predict_proba(image.reshape(1, -1))[0]
#     print("DT loaded.")
# except:
#     def dt_predict(image):
#         return np.random.dirichlet(np.ones(10))
#     print("DT not found, using stub.")

# # --- Stubs — replace with real imports when ready ---
# def nn_predict(image):
#     return np.random.dirichlet(np.ones(10))

# def attack(predict_fn, image, label, epsilon):
#     noise = np.random.uniform(-epsilon, epsilon, image.shape)
#     return np.clip(image + noise, 0, 1)

# def apply_smoothing(image, sigma):
#     return image

# # --- State ---
# current = {
#     "image": None,
#     "label": None,
#     "adversarial": None,
#     "defended": None
# }

# MNIST_CLASSES = [str(i) for i in range(10)]

# # ─────────────────────────────────────────
# # NEURAL NETWORK DIAGRAM
# # ─────────────────────────────────────────
# # Layer sizes — matches your actual NN architecture
# # Input(784) -> Dense(256) -> Dense(128) -> Output(10)
# # We display a compressed version: 6 input nodes, 6 hidden1, 5 hidden2, 4 output
# LAYER_SIZES    = [6, 6, 5, 4]
# LAYER_LABELS   = ["Input\n(784)", "Dense\n(256)", "Dense\n(128)", "Output\n(10)"]
# LAYER_COLORS   = ["#4a90d9", "#e09a3a", "#9b59b6", "#4caf7d"]
# NEURON_RADIUS  = 12
# CANVAS_W       = 420
# CANVAS_H       = 260
# ANIM_DELAY     = 60   # ms between animation frames

# def get_neuron_positions():
#     """Returns dict: layer_idx -> list of (x, y) center positions"""
#     positions = {}
#     x_gap = CANVAS_W // (len(LAYER_SIZES) + 1)
#     for li, size in enumerate(LAYER_SIZES):
#         x = x_gap * (li + 1)
#         y_gap = CANVAS_H // (size + 1)
#         positions[li] = [(x, y_gap * (ni + 1)) for ni in range(size)]
#     return positions

# NEURON_POS = get_neuron_positions()

# def draw_network(canvas, active_layer=-1, activations=None):
#     """
#     Draws the full network on the tk canvas.
#     active_layer: which layer is currently lit up (-1 = none)
#     activations: list of floats (0-1) per neuron in active layer
#     """
#     canvas.delete("all")

#     # Draw connections first (behind neurons)
#     for li in range(len(LAYER_SIZES) - 1):
#         for (x1, y1) in NEURON_POS[li]:
#             for (x2, y2) in NEURON_POS[li + 1]:
#                 # light up connections leading INTO active layer
#                 if active_layer == li + 1:
#                     color = "#f0c040"
#                     width = 1.5
#                 else:
#                     color = "#cccccc"
#                     width = 0.8
#                 canvas.create_line(x1, y1, x2, y2,
#                                    fill=color, width=width)

#     # Draw neurons
#     for li, positions in NEURON_POS.items():
#         for ni, (x, y) in enumerate(positions):
#             r = NEURON_RADIUS

#             # Pick fill color
#             if li == active_layer:
#                 if activations is not None and ni < len(activations):
#                     intensity = activations[ni]
#                     # blend from gray to layer color based on intensity
#                     fill = blend_color("#dddddd", LAYER_COLORS[li], intensity)
#                 else:
#                     fill = LAYER_COLORS[li]
#                 outline = "white"
#                 width = 2
#             else:
#                 fill = "#e8e8e8"
#                 outline = "#aaaaaa"
#                 width = 1

#             canvas.create_oval(x - r, y - r, x + r, y + r,
#                                 fill=fill, outline=outline, width=width)

#     # Draw layer labels at bottom
#     x_gap = CANVAS_W // (len(LAYER_SIZES) + 1)
#     for li, label in enumerate(LAYER_LABELS):
#         x = x_gap * (li + 1)
#         canvas.create_text(x, CANVAS_H - 8, text=label,
#                            font=("Arial", 7), fill="#555555", justify=tk.CENTER)

# def blend_color(hex1, hex2, t):
#     """Blend two hex colors. t=0 -> hex1, t=1 -> hex2"""
#     t = max(0.0, min(1.0, t))
#     r1, g1, b1 = int(hex1[1:3],16), int(hex1[3:5],16), int(hex1[5:7],16)
#     r2, g2, b2 = int(hex2[1:3],16), int(hex2[3:5],16), int(hex2[5:7],16)
#     r = int(r1 + (r2 - r1) * t)
#     g = int(g1 + (g2 - g1) * t)
#     b = int(b1 + (b2 - b1) * t)
#     return f"#{r:02x}{g:02x}{b:02x}"

# def animate_network(image):
#     """
#     Animates data flowing through the network layer by layer.
#     Uses fake activations scaled from image stats — 
#     swap with real layer activations when neural_net.py is ready.
#     """
#     # Generate fake activations per layer from image
#     # Replace these with real intermediate layer outputs later
#     fake_activations = [
#         np.abs(np.random.randn(LAYER_SIZES[0])),   # input
#         np.abs(np.random.randn(LAYER_SIZES[1])),   # dense 1
#         np.abs(np.random.randn(LAYER_SIZES[2])),   # dense 2
#         nn_predict(image)[:LAYER_SIZES[3]]          # output — real probs
#     ]

#     # Normalize each layer's activations to 0-1
#     for i in range(len(fake_activations)):
#         mx = fake_activations[i].max()
#         if mx > 0:
#             fake_activations[i] = fake_activations[i] / mx

#     def step(layer_idx):
#         if layer_idx >= len(LAYER_SIZES):
#             # Animation done — leave output layer lit
#             draw_network(nn_canvas, active_layer=len(LAYER_SIZES)-1,
#                          activations=fake_activations[-1])
#             return
#         draw_network(nn_canvas, active_layer=layer_idx,
#                      activations=fake_activations[layer_idx])
#         window.after(speed_var.get(), lambda: step(layer_idx + 1))

#     step(0)

# # ─────────────────────────────────────────
# # HELPERS
# # ─────────────────────────────────────────

# def show_image(canvas, image_array):
#     img = image_array.reshape(28, 28)
#     img = (img * 255).astype(np.uint8)
#     pil_img = Image.fromarray(img, mode='L')
#     pil_img = pil_img.resize((150, 150), Image.NEAREST)
#     photo = ImageTk.PhotoImage(pil_img)
#     canvas.configure(image=photo)
#     canvas.image = photo

# def update_predictions(image, dt_label, nn_label, true_label):
#     dt_probs = dt_predict(image)
#     dt_pred  = np.argmax(dt_probs)
#     dt_conf  = int(dt_probs[dt_pred] * 100)
#     dt_label.config(
#         text=f"DT: {dt_pred} ({dt_conf}%)",
#         fg="green" if dt_pred == true_label else "red"
#     )

#     nn_probs = nn_predict(image)
#     nn_pred  = np.argmax(nn_probs)
#     nn_conf  = int(nn_probs[nn_pred] * 100)
#     nn_label.config(
#         text=f"NN: {nn_pred} ({nn_conf}%)",
#         fg="green" if nn_pred == true_label else "red"
#     )
#     window.update()
# '''
# def update_histogram():
#     ax.clear()
#     ax.set_facecolor("#f5f5f5")
#     ax.set_xlabel("Pixel Intensity", fontsize=8)
#     ax.set_ylabel("Count", fontsize=8)
#     ax.tick_params(labelsize=7)

#     if current["image"] is not None:
#         ax.hist(current["image"], bins=50, alpha=0.6,
#                 color="steelblue", label="Original")
#     if current["adversarial"] is not None:
#         ax.hist(current["adversarial"], bins=50, alpha=0.6,
#                 color="tomato", label="After Attack")
#     if current["defended"] is not None:
#         ax.hist(current["defended"], bins=50, alpha=0.6,
#                 color="mediumseagreen", label="After Defense")

#     ax.legend(fontsize=7)
#     fig.tight_layout()
#     hist_canvas.draw()
# '''
# # ─────────────────────────────────────────
# # BUTTON FUNCTIONS
# # ─────────────────────────────────────────

# def load_image():
#     idx = np.random.randint(0, len(X_test))
#     current["image"]       = X_test[idx]
#     current["label"]       = int(y_test[idx])
#     current["adversarial"] = None
#     current["defended"]    = None

#     show_image(original_canvas, current["image"])
#     update_predictions(current["image"],
#                        orig_dt_label, orig_nn_label, current["label"])
#     true_label_display.config(text=f"True Label: {current['label']}")

#     attack_canvas.configure(image='')
#     defense_canvas.configure(image='')
#     for lbl in [attack_dt_label, attack_nn_label,
#                 defense_dt_label, defense_nn_label]:
#         lbl.config(text="—", fg="black")

#     #update_histogram()
#     animate_network(current["image"])
#     status_label.config(text="Image loaded. Ready to attack.")

# def run_attack():
#     if current["image"] is None:
#         status_label.config(text="Load an image first.")
#         return
#     status_label.config(text="Running attack... please wait.")
#     window.update()

#     adv = attack(dt_predict, current["image"],
#                  current["label"], epsilon_var.get())
#     current["adversarial"] = adv

#     show_image(attack_canvas, adv)
#     update_predictions(adv, attack_dt_label, attack_nn_label, current["label"])
#     #update_histogram()
#     animate_network(adv)
#     status_label.config(text="Attack done. Red = fooled, Green = survived.")

# def run_defense():
#     if current["adversarial"] is None:
#         status_label.config(text="Run attack first.")
#         return
#     status_label.config(text="Applying defense...")
#     window.update()

#     defended = apply_smoothing(current["adversarial"], sigma_var.get())
#     current["defended"] = defended

#     show_image(defense_canvas, defended)
#     update_predictions(defended, defense_dt_label,
#                        defense_nn_label, current["label"])
#     #update_histogram()
#     animate_network(defended)
#     status_label.config(text="Defense done. Green = recovered, Red = still fooled.")

# def reset():
#     current.update({"image": None, "label": None,
#                     "adversarial": None, "defended": None})
#     for canvas in [original_canvas, attack_canvas, defense_canvas]:
#         canvas.configure(image='')
#     for lbl in [orig_dt_label, orig_nn_label,
#                 attack_dt_label, attack_nn_label,
#                 defense_dt_label, defense_nn_label]:
#         lbl.config(text="—", fg="black")
#     true_label_display.config(text="True Label: —")
#     draw_network(nn_canvas)
#     #ax.clear()
#     #ax.set_title("Load an image to begin", fontsize=8)
#     #hist_canvas.draw()
#     status_label.config(text="Reset.")

# # ─────────────────────────────────────────
# # WINDOW
# # ─────────────────────────────────────────

# window = tk.Tk()
# window.title("Adversarial Attack & Defense Demo")
# window.geometry("1200x780")
# window.configure(bg="#E3F6EC")

# # ─────────────────────────────────────────
# # LEFT PANEL — controls
# # ─────────────────────────────────────────

# left_panel = tk.Frame(window, bg="#E3F6EC", width=175)
# left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=12, pady=12)

# tk.Label(left_panel, text="Controls",
#          font=("Arial", 12, "bold"), bg="#E3F6EC").pack(pady=8)

# tk.Label(left_panel, text="Epsilon:",
#          bg="#E3F6EC", font=("Arial", 9)).pack(pady=(12, 0))
# epsilon_var = tk.DoubleVar(value=0.1)
# tk.Scale(left_panel, from_=0.05, to=0.25, resolution=0.05,
#          orient=tk.HORIZONTAL, variable=epsilon_var,
#          bg="#E3F6EC", length=150).pack()

# tk.Label(left_panel, text="Sigma:",
#          bg="#E3F6EC", font=("Arial", 9)).pack(pady=(12, 0))
# sigma_var = tk.DoubleVar(value=1.0)
# tk.Scale(left_panel, from_=0.5, to=2.0, resolution=0.5,
#          orient=tk.HORIZONTAL, variable=sigma_var,
#          bg="#E3F6EC", length=150).pack()

# true_label_display = tk.Label(left_panel, text="True Label: —",
#                                bg="#E3F6EC", font=("Arial", 10, "bold"))
# true_label_display.pack(pady=(18, 5))

# tk.Button(left_panel, text="Load Random Image",
#           command=load_image, width=17,
#           bg="#4a90d9", fg="white", relief=tk.FLAT).pack(pady=(8, 4))
# tk.Button(left_panel, text="Run Attack",
#           command=run_attack, width=17,
#           bg="#e05555", fg="white", relief=tk.FLAT).pack(pady=4)
# tk.Button(left_panel, text="Apply Defense",
#           command=run_defense, width=17,
#           bg="#4caf7d", fg="white", relief=tk.FLAT).pack(pady=4)
# tk.Button(left_panel, text="Reset",
#           command=reset, width=17,
#           bg="#aaaaaa", fg="white", relief=tk.FLAT).pack(pady=4)

# # ─────────────────────────────────────────
# # RIGHT AREA
# # ─────────────────────────────────────────

# right_area = tk.Frame(window, bg="#E3F6EC")
# right_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=12, padx=8)

# # ── TOP: images ──
# image_panel = tk.Frame(right_area, bg="#E3F6EC")
# image_panel.pack(fill=tk.X)

# for col, title in enumerate(["Original", "After Attack", "After Defense"]):
#     tk.Label(image_panel, text=title, bg="#E3F6EC",
#              font=("Arial", 10, "bold")).grid(row=0, column=col, padx=20)

# original_canvas = tk.Label(image_panel, bg="white", width=150, height=150)
# original_canvas.grid(row=1, column=0, padx=20, pady=6)
# attack_canvas   = tk.Label(image_panel, bg="white", width=150, height=150)
# attack_canvas.grid(row=1, column=1, padx=20, pady=6)
# defense_canvas  = tk.Label(image_panel, bg="white", width=150, height=150)
# defense_canvas.grid(row=1, column=2, padx=20, pady=6)

# orig_dt_label    = tk.Label(image_panel, text="DT: —", bg="#E3F6EC", font=("Arial", 9))
# orig_dt_label.grid(row=2, column=0)
# attack_dt_label  = tk.Label(image_panel, text="DT: —", bg="#E3F6EC", font=("Arial", 9))
# attack_dt_label.grid(row=2, column=1)
# defense_dt_label = tk.Label(image_panel, text="DT: —", bg="#E3F6EC", font=("Arial", 9))
# defense_dt_label.grid(row=2, column=2)

# orig_nn_label    = tk.Label(image_panel, text="NN: —", bg="#E3F6EC", font=("Arial", 9))
# orig_nn_label.grid(row=3, column=0)
# attack_nn_label  = tk.Label(image_panel, text="NN: —", bg="#E3F6EC", font=("Arial", 9))
# attack_nn_label.grid(row=3, column=1)
# defense_nn_label = tk.Label(image_panel, text="NN: —", bg="#E3F6EC", font=("Arial", 9))
# defense_nn_label.grid(row=3, column=2)

# # ── BOTTOM: NN diagram + histogram side by side ──
# bottom_area = tk.Frame(right_area, bg="#E3F6EC")
# bottom_area.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

# # NN canvas on the left
# nn_frame = tk.Frame(bottom_area, bg="#E3F6EC")
# nn_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))

# tk.Label(nn_frame, text="Neural Network",
#          bg="#E3F6EC", font=("Arial", 9, "bold")).pack()
# nn_canvas = tk.Canvas(nn_frame, width=CANVAS_W, height=CANVAS_H,
#                       bg="#f8f8f8", highlightthickness=1,
#                       highlightbackground="#cccccc")
# nn_canvas.pack()

# # Controls under the NN diagram
# nn_controls = tk.Frame(nn_frame, bg="#E3F6EC")
# nn_controls.pack(fill=tk.X, pady=(4, 0))

# tk.Label(nn_controls, text="Speed:",
#          bg="#E3F6EC", font=("Arial", 8)).pack(side=tk.LEFT, padx=(0, 4))
# speed_var = tk.IntVar(value=60)
# tk.Scale(nn_controls, from_=10, to=300, resolution=10,
#          orient=tk.HORIZONTAL, variable=speed_var,
#          bg="#E3F6EC", length=150,
#          label="fast         slow").pack(side=tk.LEFT)

# tk.Button(nn_controls, text="Replay",
#           command=lambda: animate_network(current["image"]) if current["image"] is not None else None,
#           bg="#4a90d9", fg="white", relief=tk.FLAT,
#           font=("Arial", 8), width=8).pack(side=tk.LEFT, padx=(10, 0))

# # Histogram on the right
# '''
# hist_frame = tk.Frame(bottom_area, bg="#E3F6EC")
# hist_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# tk.Label(hist_frame, text="Pixel Intensity Distribution",
#          bg="#E3F6EC", font=("Arial", 9, "bold")).pack()

# fig, ax = plt.subplots(figsize=(5, 2.8))
# fig.patch.set_facecolor("#E3F6EC")
# ax.set_facecolor("#f5f5f5")
# ax.set_title("Load an image to begin", fontsize=8)

# hist_canvas = FigureCanvasTkAgg(fig, master=hist_frame)
# hist_canvas.draw()
# hist_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
# '''
# # ─────────────────────────────────────────
# # STATUS BAR
# # ─────────────────────────────────────────

# status_label = tk.Label(window, text="Run mnist.py first, then launch this.",
#                         bg="#dddddd", anchor="w", font=("Arial", 9))
# status_label.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=3)

# # ── Draw empty network on startup ──
# draw_network(nn_canvas)

# window.mainloop()
import tkinter as tk
from tkinter import font as tkfont
from PIL import Image, ImageTk
import numpy as np
import pickle
import time
from keras.models import load_model


#AUTO SETUP 
import setup    #setup.py handles mnist.py + DT + NN training

# --- Load data ---
nn_model = load_model('nn_model.keras')

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

# def nn_predict(image):
#     return np.random.dirichlet(np.ones(10))
def nn_predict(image):
    return nn_model.predict(image.reshape(1, -1), verbose=0)[0]

# def attack(predict_fn, image, label, epsilon):
#     noise = np.random.uniform(-epsilon, epsilon, image.shape)
#     return np.clip(image + noise, 0, 1)
from genetic_attack import GeneticAttack

# def attack(predict_fn, image, label, epsilon):
#     ga = GeneticAttack(epsilon=epsilon, pop_size=50, n_generations=100)
#     adv, success, gen = ga.attack(image, label, predict_fn, predict_proba_fn)
#     return adv

def predict_proba_fn(adv_batch):
    # return nn_model.predict(image.reshape(1, -1), verbose=0)
    return nn_model.predict(adv_batch, verbose=0)

def attack(predict_fn, image, label, epsilon):
    ga = GeneticAttack(epsilon=epsilon, pop_size=100, n_generations=200)
    adv, success, gen = ga.attack(image, label, predict_fn, predict_proba_fn)
    print(f"Attack - Success: {success}, Generation: {gen}")
    return adv

# def apply_smoothing(image, sigma):
#     return image
from defense import EnhancedDefense

# def apply_smoothing(image, sigma):
#     return EnhancedDefense(image, size=3, sigma=sigma)
def apply_smoothing(image, sigma):
    """Apply enhanced defense to a single image"""
    defense = EnhancedDefense()
    return defense.defend(image, method='ensemble', sigma=sigma)

# --- State ---
current = {
    "image": None,
    "label": None,
    "adversarial": None,
    "defended": None
}

# ─────────────────────────────────────────
# COLOR PALETTE — dark lab theme
# ─────────────────────────────────────────
BG          = "#0d0f14"
PANEL_BG    = "#13161e"
CARD_BG     = "#1a1d28"
BORDER      = "#2a2d3e"
ACCENT_BLUE = "#4f8ef7"
ACCENT_RED  = "#f74f6a"
ACCENT_GRN  = "#4ff7a0"
ACCENT_YLW  = "#f7c84f"
TEXT_PRI    = "#eef0f8"
TEXT_SEC    = "#7a7f9a"
TEXT_DIM    = "#3a3f55"

# NN diagram
LAYER_SIZES  = [6, 6, 5, 4]
LAYER_LABELS = ["Input\n784", "Dense\n256", "Dense\n128", "Output\n10"]
LAYER_COLORS = [ACCENT_BLUE, ACCENT_YLW, "#b44ff7", ACCENT_GRN]
NEURON_R     = 11
CANVAS_W     = 400
CANVAS_H     = 220

def get_neuron_positions():
    positions = {}
    x_gap = CANVAS_W // (len(LAYER_SIZES) + 1)
    for li, size in enumerate(LAYER_SIZES):
        x = x_gap * (li + 1)
        y_gap = CANVAS_H // (size + 1)
        positions[li] = [(x, y_gap * (ni + 1)) for ni in range(size)]
    return positions

NEURON_POS = get_neuron_positions()

def blend_color(hex1, hex2, t):
    t = max(0.0, min(1.0, t))
    r1,g1,b1 = int(hex1[1:3],16),int(hex1[3:5],16),int(hex1[5:7],16)
    r2,g2,b2 = int(hex2[1:3],16),int(hex2[3:5],16),int(hex2[5:7],16)
    r = int(r1+(r2-r1)*t)
    g = int(g1+(g2-g1)*t)
    b = int(b1+(b2-b1)*t)
    return f"#{r:02x}{g:02x}{b:02x}"

def draw_network(canvas, active_layer=-1, activations=None):
    canvas.delete("all")
    # connections
    for li in range(len(LAYER_SIZES)-1):
        for (x1,y1) in NEURON_POS[li]:
            for (x2,y2) in NEURON_POS[li+1]:
                if active_layer == li+1:
                    color = blend_color(LAYER_COLORS[li], LAYER_COLORS[li+1], 0.5)
                    width = 1.5
                else:
                    color = TEXT_DIM
                    width = 0.6
                canvas.create_line(x1,y1,x2,y2,fill=color,width=width)
    # neurons
    for li, positions in NEURON_POS.items():
        for ni,(x,y) in enumerate(positions):
            r = NEURON_R
            if li == active_layer:
                if activations is not None and ni < len(activations):
                    intensity = activations[ni]
                    fill = blend_color(CARD_BG, LAYER_COLORS[li], intensity)
                else:
                    fill = LAYER_COLORS[li]
                outline = LAYER_COLORS[li]
                width = 2
            else:
                fill = CARD_BG
                outline = BORDER
                width = 1
            canvas.create_oval(x-r,y-r,x+r,y+r,fill=fill,outline=outline,width=width)
    # labels
    x_gap = CANVAS_W//(len(LAYER_SIZES)+1)
    for li,label in enumerate(LAYER_LABELS):
        x = x_gap*(li+1)
        canvas.create_text(x, CANVAS_H-10, text=label,
                           font=("Courier",7), fill=TEXT_SEC, justify=tk.CENTER)

def animate_network(image):
    fake_activations = [
        np.abs(np.random.randn(LAYER_SIZES[0])),
        np.abs(np.random.randn(LAYER_SIZES[1])),
        np.abs(np.random.randn(LAYER_SIZES[2])),
        nn_predict(image)[:LAYER_SIZES[3]]
    ]
    for i in range(len(fake_activations)):
        mx = fake_activations[i].max()
        if mx > 0:
            fake_activations[i] = fake_activations[i] / mx

    def step(layer_idx):
        if layer_idx >= len(LAYER_SIZES):
            draw_network(nn_canvas, active_layer=len(LAYER_SIZES)-1,
                         activations=fake_activations[-1])
            return
        draw_network(nn_canvas, active_layer=layer_idx,
                     activations=fake_activations[layer_idx])
        window.after(speed_var.get(), lambda: step(layer_idx+1))
    step(0)

# ─────────────────────────────────────────
# IMAGE DISPLAY
# ─────────────────────────────────────────

def show_image(label_widget, image_array):
    img = image_array.reshape(28,28)
    img = (img*255).astype(np.uint8)
    pil_img = Image.fromarray(img, mode='L').resize((140,140), Image.NEAREST)
    photo = ImageTk.PhotoImage(pil_img)
    label_widget.configure(image=photo)
    label_widget.image = photo

def make_confidence_bars(frame, probs):
    for widget in frame.winfo_children():
        widget.destroy()
    for i, p in enumerate(probs):
        row = tk.Frame(frame, bg=CARD_BG)
        row.pack(fill=tk.X, pady=1)
        tk.Label(row, text=str(i), width=2, bg=CARD_BG,
                 fg=TEXT_SEC, font=("Courier",7)).pack(side=tk.LEFT)
        bar_bg = tk.Frame(row, bg=BORDER, height=8, width=100)
        bar_bg.pack(side=tk.LEFT, padx=3)
        bar_bg.pack_propagate(False)
        fill_w = int(p*100)
        color = ACCENT_GRN if i == np.argmax(probs) else ACCENT_BLUE
        tk.Frame(bar_bg, bg=color, height=8, width=fill_w).place(x=0,y=0)
        tk.Label(row, text=f"{p:.2f}", bg=CARD_BG,
                 fg=TEXT_DIM, font=("Courier",7)).pack(side=tk.LEFT)

def update_card(img_label, dt_bar_frame, nn_bar_frame,
                pred_label, true_label_val, image):
    show_image(img_label, image)
    dt_probs = dt_predict(image)
    nn_probs = nn_predict(image)
    dt_pred  = np.argmax(dt_probs)
    nn_pred  = np.argmax(nn_probs)
    make_confidence_bars(dt_bar_frame, dt_probs)
    make_confidence_bars(nn_bar_frame, nn_probs)
    dt_col = ACCENT_GRN if dt_pred == true_label_val else ACCENT_RED
    nn_col = ACCENT_GRN if nn_pred == true_label_val else ACCENT_RED
    # pred_label.config(
    #     text=f"DT → {dt_pred}   NN → {nn_pred}",
    #     fg=ACCENT_GRN if dt_pred == true_label_val and nn_pred == true_label_val else ACCENT_RED
    # )
    dt_conf = int(dt_probs[dt_pred] * 100)
    nn_conf = int(nn_probs[nn_pred] * 100)
    dt_col = ACCENT_GRN if dt_pred == true_label_val else ACCENT_RED
    nn_col = ACCENT_GRN if nn_pred == true_label_val else ACCENT_RED
    pred_label.config(
    text=f"DT predicts: {dt_pred}  ({dt_conf}%)\nNN predicts: {nn_pred}  ({nn_conf}%)",
    fg=dt_col
)

# ─────────────────────────────────────────
# BUTTON ACTIONS
# ─────────────────────────────────────────

def load_image():
    idx = np.random.randint(0, len(X_test))
    current["image"]       = X_test[idx]
    current["label"]       = int(y_test[idx])
    current["adversarial"] = None
    current["defended"]    = None

    update_card(orig_img, orig_dt_bars, orig_nn_bars,
                orig_pred_lbl, current["label"], current["image"])

    for w in [atk_img, def_img]:
        w.configure(image='')
    for f in [atk_dt_bars, atk_nn_bars, def_dt_bars, def_nn_bars]:
        for c in f.winfo_children(): c.destroy()
    for l in [atk_pred_lbl, def_pred_lbl]:
        l.config(text="—", fg=TEXT_SEC)

    true_lbl.config(text=f"TRUE LABEL:  {current['label']}")
    status_lbl.config(text="▶️  Image loaded. Ready to attack.", fg=ACCENT_BLUE)
    animate_network(current["image"])

def run_attack():
    if current["image"] is None:
        status_lbl.config(text="⚠  Load an image first.", fg=ACCENT_YLW)
        return
    status_lbl.config(text="⚡  Running genetic attack...", fg=ACCENT_YLW)
    window.update()
    adv = attack(dt_predict, current["image"], current["label"], epsilon_var.get())
    current["adversarial"] = adv
    update_card(atk_img, atk_dt_bars, atk_nn_bars,
                atk_pred_lbl, current["label"], adv)
    status_lbl.config(text="✗  Attack complete. Red = fooled.", fg=ACCENT_RED)
    animate_network(adv)

def run_defense():
    if current["adversarial"] is None:
        status_lbl.config(text="⚠  Run attack first.", fg=ACCENT_YLW)
        return
    status_lbl.config(text="🛡  Applying Gaussian defense...", fg=ACCENT_GRN)
    window.update()
    defended = apply_smoothing(current["adversarial"], sigma_var.get())
    current["defended"] = defended
    update_card(def_img, def_dt_bars, def_nn_bars,
                def_pred_lbl, current["label"], defended)
    status_lbl.config(text="✔  Defense applied. Green = recovered.", fg=ACCENT_GRN)
    animate_network(defended)

def reset_all():
    current.update({"image":None,"label":None,"adversarial":None,"defended":None})
    for w in [orig_img, atk_img, def_img]:
        w.configure(image='')
    for f in [orig_dt_bars,orig_nn_bars,atk_dt_bars,atk_nn_bars,def_dt_bars,def_nn_bars]:
        for c in f.winfo_children(): c.destroy()
    for l in [orig_pred_lbl, atk_pred_lbl, def_pred_lbl]:
        l.config(text="—", fg=TEXT_SEC)
    true_lbl.config(text="TRUE LABEL:  —")
    draw_network(nn_canvas)
    status_lbl.config(text="↺  Reset.", fg=TEXT_SEC)

# ─────────────────────────────────────────
# WINDOW SETUP
# ─────────────────────────────────────────

window = tk.Tk()
window.title("Adversarial Robustness — Attack & Defense Demo")
window.geometry("1280x800")
window.configure(bg=BG)
window.resizable(False, False)

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────

header = tk.Frame(window, bg=PANEL_BG, height=52)
header.pack(fill=tk.X)
header.pack_propagate(False)

tk.Label(header, text="ADVERSARIAL ROBUSTNESS",
         bg=PANEL_BG, fg=TEXT_PRI,
         font=("Courier",15,"bold")).pack(side=tk.LEFT, padx=20, pady=14)

tk.Label(header, text="MNIST · Decision Tree · Neural Network · Genetic Attack · Gaussian Defense",
         bg=PANEL_BG, fg=TEXT_SEC,
         font=("Courier",8)).pack(side=tk.LEFT, padx=0, pady=14)

true_lbl = tk.Label(header, text="TRUE LABEL:  —",
                    bg=PANEL_BG, fg=ACCENT_YLW,
                    font=("Courier",11,"bold"))
true_lbl.pack(side=tk.RIGHT, padx=20)

# thin accent line under header
tk.Frame(window, bg=ACCENT_BLUE, height=2).pack(fill=tk.X)

# ─────────────────────────────────────────
# MAIN BODY
# ─────────────────────────────────────────

body = tk.Frame(window, bg=BG)
body.pack(fill=tk.BOTH, expand=True, padx=14, pady=10)

# ── LEFT SIDEBAR ──
sidebar = tk.Frame(body, bg=PANEL_BG, width=170)
sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0,12))
sidebar.pack_propagate(False)

def section_label(parent, text):
    tk.Label(parent, text=text, bg=PANEL_BG, fg=TEXT_SEC,
             font=("Courier",7,"bold")).pack(anchor="w", padx=14, pady=(14,2))

def styled_btn(parent, text, cmd, color):
    tk.Button(parent, text=text, command=cmd,
              bg=color, fg=BG, activebackground=color,
              relief=tk.FLAT, font=("Courier",9,"bold"),
              cursor="hand2", pady=7, width=16).pack(padx=12, pady=3)

tk.Label(sidebar, text="CONTROLS", bg=PANEL_BG, fg=TEXT_PRI,
         font=("Courier",11,"bold")).pack(pady=(18,4))
tk.Frame(sidebar, bg=BORDER, height=1).pack(fill=tk.X, padx=12)

section_label(sidebar, "EPSILON  (attack strength)")
epsilon_var = tk.DoubleVar(value=0.3)
tk.Scale(sidebar, from_=0.1, to=0.5, resolution=0.05,
         orient=tk.HORIZONTAL, variable=epsilon_var,
         bg=PANEL_BG, fg=TEXT_PRI, troughcolor=BORDER,
         highlightthickness=0, length=145,
         font=("Courier",7)).pack(padx=12)

section_label(sidebar, "SIGMA  (defense strength)")
sigma_var = tk.DoubleVar(value=2.0)
tk.Scale(sidebar, from_=0.5, to=3.0, resolution=0.5,
         orient=tk.HORIZONTAL, variable=sigma_var,
         bg=PANEL_BG, fg=TEXT_PRI, troughcolor=BORDER,
         highlightthickness=0, length=145,
         font=("Courier",7)).pack(padx=12)

section_label(sidebar, "ANIM SPEED")
speed_var = tk.IntVar(value=80)
tk.Scale(sidebar, from_=20, to=300, resolution=20,
         orient=tk.HORIZONTAL, variable=speed_var,
         bg=PANEL_BG, fg=TEXT_PRI, troughcolor=BORDER,
         highlightthickness=0, length=145,
         font=("Courier",7)).pack(padx=12)

tk.Frame(sidebar, bg=BORDER, height=1).pack(fill=tk.X, padx=12, pady=10)

styled_btn(sidebar, "LOAD IMAGE",    load_image,  ACCENT_BLUE)
styled_btn(sidebar, "RUN ATTACK",    run_attack,  ACCENT_RED)
styled_btn(sidebar, "APPLY DEFENSE", run_defense, ACCENT_GRN)
styled_btn(sidebar, "RESET",         reset_all,   TEXT_DIM)

# NN diagram at bottom of sidebar
tk.Frame(sidebar, bg=BORDER, height=1).pack(fill=tk.X, padx=12, pady=(14,6))
tk.Label(sidebar, text="NETWORK", bg=PANEL_BG, fg=TEXT_SEC,
         font=("Courier",7,"bold")).pack()

nn_canvas = tk.Canvas(sidebar, width=160, height=160,
                      bg=CARD_BG, highlightthickness=0)
nn_canvas.pack(padx=5, pady=4)

# Override CANVAS_W/H for sidebar nn diagram
CANVAS_W_ORIG, CANVAS_H_ORIG = CANVAS_W, CANVAS_H
CANVAS_W, CANVAS_H = 160, 160
NEURON_POS = get_neuron_positions()
CANVAS_W, CANVAS_H = CANVAS_W_ORIG, CANVAS_H_ORIG

tk.Button(sidebar, text="REPLAY ANIM",
          command=lambda: animate_network(current["image"]) if current["image"] else None,
          bg=BORDER, fg=TEXT_SEC, relief=tk.FLAT,
          font=("Courier",7), cursor="hand2").pack(pady=4)

# ── RIGHT CONTENT — 3 cards ──
content = tk.Frame(body, bg=BG)
content.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

def make_card(parent, title, accent):
    card = tk.Frame(parent, bg=CARD_BG, bd=0)
    card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

    # title bar
    title_bar = tk.Frame(card, bg=accent, height=3)
    title_bar.pack(fill=tk.X)

    tk.Label(card, text=title, bg=CARD_BG, fg=accent,
             font=("Courier",10,"bold")).pack(pady=(8,4))

    # image
    img_lbl = tk.Label(card, bg="#0a0c10", width=140, height=140)
    img_lbl.pack(pady=4, padx=10)

    # prediction line
    pred_lbl = tk.Label(card, text="—", bg=CARD_BG, fg=TEXT_SEC,
                        font=("Courier",11,"bold"))
    pred_lbl.pack(pady=2)

    # confidence bars
    tk.Label(card, text="DT CONFIDENCE", bg=CARD_BG, fg=TEXT_DIM,
             font=("Courier",6)).pack(anchor="w", padx=10, pady=(6,0))
    dt_bars = tk.Frame(card, bg=CARD_BG)
    dt_bars.pack(fill=tk.X, padx=10)

    tk.Label(card, text="NN CONFIDENCE", bg=CARD_BG, fg=TEXT_DIM,
             font=("Courier",6)).pack(anchor="w", padx=10, pady=(6,0))
    nn_bars = tk.Frame(card, bg=CARD_BG)
    nn_bars.pack(fill=tk.X, padx=10, pady=(0,8))

    return img_lbl, pred_lbl, dt_bars, nn_bars

orig_img, orig_pred_lbl, orig_dt_bars, orig_nn_bars = make_card(content, "ORIGINAL", ACCENT_BLUE)
atk_img,  atk_pred_lbl,  atk_dt_bars,  atk_nn_bars  = make_card(content, "AFTER ATTACK", ACCENT_RED)
def_img,  def_pred_lbl,  def_dt_bars,  def_nn_bars  = make_card(content, "AFTER DEFENSE", ACCENT_GRN)

# ─────────────────────────────────────────
# STATUS BAR
# ─────────────────────────────────────────

tk.Frame(window, bg=BORDER, height=1).pack(fill=tk.X)
status_bar = tk.Frame(window, bg=PANEL_BG, height=30)
status_bar.pack(fill=tk.X)
status_bar.pack_propagate(False)

status_lbl = tk.Label(status_bar,
                      text="▶️  Run data_extraction.py first, then launch this.",
                      bg=PANEL_BG, fg=TEXT_SEC, font=("Courier",8), anchor="w")
status_lbl.pack(side=tk.LEFT, padx=16, pady=6)

tk.Label(status_bar, text="Group: Ayesha · Arwa · Eshaal",
         bg=PANEL_BG, fg=TEXT_DIM, font=("Courier",7)).pack(side=tk.RIGHT, padx=16)

# ─────────────────────────────────────────
# INIT
# ─────────────────────────────────────────
draw_network(nn_canvas)
window.mainloop()