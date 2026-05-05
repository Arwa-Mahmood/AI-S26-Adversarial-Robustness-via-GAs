import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import pickle
from keras.models import load_model
from genetic_attack import GeneticAttack
from defense import apply_gaussian_smoothing

# --- Load models and data ---
nn_model = load_model('nn_model.keras')
X_test = np.load('x_test_final_data.npy')
y_test = np.load('y_test_final_data.npy')

try:
    with open('models/decision_tree.pkl', 'rb') as f:
        dt_model = pickle.load(f)
    def dt_predict(image):
        return dt_model.predict_proba(image.reshape(1, -1))[0]
    print("DT loaded.")
except Exception as e:
    print(f"DT error: {e}")
    def dt_predict(image):
        return np.random.dirichlet(np.ones(10))

def nn_predict(image):
    return nn_model.predict(image.reshape(1, -1), verbose=0)[0]

def predict_proba_fn(adv_batch):
    return nn_model.predict(adv_batch, verbose=0)

def attack(image, label, epsilon):
    ga = GeneticAttack(epsilon=epsilon, pop_size=30, n_generations=30)
    adv, success, gen = ga.attack(image, label,
                                   lambda x: np.array([np.argmax(dt_predict(x[0]))]),
                                   predict_proba_fn)
    return adv

def apply_smoothing(image, sigma):
    return apply_gaussian_smoothing(image, size=3, sigma=sigma)

# --- State ---
current = {"image": None, "label": None, "adversarial": None, "defended": None}

# --- Colors ---
BG       = "#0d0f14"
PANEL_BG = "#13161e"
CARD_BG  = "#1a1d28"
BORDER   = "#2a2d3e"
A_BLUE   = "#4f8ef7"
A_RED    = "#f74f6a"
A_GRN    = "#4ff7a0"
A_YLW    = "#f7c84f"
T_PRI    = "#eef0f8"
T_SEC    = "#7a7f9a"
T_DIM    = "#3a3f55"

# --- NN diagram ---
LAYER_SIZES  = [6, 6, 5, 4]
LAYER_LABELS = ["Input\n784", "Dense\n256", "Dense\n128", "Output\n10"]
LAYER_COLORS = [A_BLUE, A_YLW, "#b44ff7", A_GRN]
CANVAS_W, CANVAS_H = 160, 160
NEURON_R = 10

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
    return f"#{int(r1+(r2-r1)*t):02x}{int(g1+(g2-g1)*t):02x}{int(b1+(b2-b1)*t):02x}"

def draw_network(canvas, active_layer=-1, activations=None):
    canvas.delete("all")
    for li in range(len(LAYER_SIZES)-1):
        for (x1,y1) in NEURON_POS[li]:
            for (x2,y2) in NEURON_POS[li+1]:
                color = blend_color(LAYER_COLORS[li], LAYER_COLORS[li+1], 0.5) if active_layer == li+1 else T_DIM
                canvas.create_line(x1,y1,x2,y2,fill=color,width=1.5 if active_layer==li+1 else 0.6)
    for li, positions in NEURON_POS.items():
        for ni,(x,y) in enumerate(positions):
            r = NEURON_R
            if li == active_layer:
                intensity = activations[ni] if activations is not None and ni < len(activations) else 1.0
                fill = blend_color(CARD_BG, LAYER_COLORS[li], intensity)
                outline, width = LAYER_COLORS[li], 2
            else:
                fill, outline, width = CARD_BG, BORDER, 1
            canvas.create_oval(x-r,y-r,x+r,y+r,fill=fill,outline=outline,width=width)
    x_gap = CANVAS_W//(len(LAYER_SIZES)+1)
    for li,label in enumerate(LAYER_LABELS):
        canvas.create_text(x_gap*(li+1), CANVAS_H-10, text=label, font=("Courier",6), fill=T_SEC, justify=tk.CENTER)

def animate_network(image):
    fake = [np.abs(np.random.randn(LAYER_SIZES[i])) for i in range(3)]
    fake.append(nn_predict(image)[:LAYER_SIZES[3]])
    for i in range(len(fake)):
        mx = fake[i].max()
        if mx > 0: fake[i] = fake[i] / mx

    def step(li):
        if li >= len(LAYER_SIZES):
            draw_network(nn_canvas, active_layer=len(LAYER_SIZES)-1, activations=fake[-1])
            return
        draw_network(nn_canvas, active_layer=li, activations=fake[li])
        window.after(speed_var.get(), lambda: step(li+1))
    step(0)

# --- Image display ---
def show_image(lbl, arr):
    img = (arr.reshape(28,28)*255).astype(np.uint8)
    pil = Image.fromarray(img, mode='L').resize((160,160), Image.NEAREST)
    photo = ImageTk.PhotoImage(pil)
    lbl.configure(image=photo)
    lbl.image = photo

def update_card(img_lbl, pred_lbl, true_val, image):
    show_image(img_lbl, image)
    dt_probs = dt_predict(image)
    nn_probs = nn_predict(image)
    dt_pred  = int(np.argmax(dt_probs))
    nn_pred  = int(np.argmax(nn_probs))
    dt_conf  = int(dt_probs[dt_pred] * 100)
    nn_conf  = int(nn_probs[nn_pred] * 100)
    both_correct = (dt_pred == true_val and nn_pred == true_val)
    pred_lbl.config(
        text=f"DT predicts: {dt_pred}  ({dt_conf}%)\nNN predicts: {nn_pred}  ({nn_conf}%)",
        fg=A_GRN if both_correct else A_RED
    )

# --- Button actions ---
def load_image():
    idx = np.random.randint(0, len(X_test))
    current["image"]       = X_test[idx]
    current["label"]       = int(y_test[idx])
    current["adversarial"] = None
    current["defended"]    = None
    update_card(orig_img, orig_pred_lbl, current["label"], current["image"])
    atk_img.configure(image='')
    def_img.configure(image='')
    atk_pred_lbl.config(text="—", fg=T_SEC)
    def_pred_lbl.config(text="—", fg=T_SEC)
    true_lbl.config(text=f"TRUE LABEL:  {current['label']}")
    status_lbl.config(text="Image loaded. Ready to attack.", fg=A_BLUE)
    animate_network(current["image"])

def run_attack():
    if current["image"] is None:
        status_lbl.config(text="Load an image first.", fg=A_YLW)
        return
    status_lbl.config(text="Running genetic attack... please wait.", fg=A_YLW)
    window.update()
    adv = attack(current["image"], current["label"], epsilon_var.get())
    current["adversarial"] = adv
    update_card(atk_img, atk_pred_lbl, current["label"], adv)
    status_lbl.config(text="Attack complete. Red = fooled.", fg=A_RED)
    animate_network(adv)

def run_defense():
    if current["adversarial"] is None:
        status_lbl.config(text="Run attack first.", fg=A_YLW)
        return
    status_lbl.config(text="Applying Gaussian defense...", fg=A_GRN)
    window.update()
    defended = apply_smoothing(current["adversarial"], sigma_var.get())
    current["defended"] = defended
    update_card(def_img, def_pred_lbl, current["label"], defended)
    status_lbl.config(text="Defense applied. Green = recovered.", fg=A_GRN)
    animate_network(defended)

def reset_all():
    current.update({"image":None,"label":None,"adversarial":None,"defended":None})
    for w in [orig_img, atk_img, def_img]:
        w.configure(image='')
    for l in [orig_pred_lbl, atk_pred_lbl, def_pred_lbl]:
        l.config(text="—", fg=T_SEC)
    true_lbl.config(text="TRUE LABEL:  —")
    draw_network(nn_canvas)
    status_lbl.config(text="Reset.", fg=T_SEC)

# --- Window ---
window = tk.Tk()
window.title("Adversarial Robustness — Attack & Defense Demo")
window.geometry("1280x780")
window.configure(bg=BG)
window.resizable(False, False)

# Header
header = tk.Frame(window, bg=PANEL_BG, height=52)
header.pack(fill=tk.X)
header.pack_propagate(False)
tk.Label(header, text="ADVERSARIAL ROBUSTNESS", bg=PANEL_BG, fg=T_PRI,
         font=("Courier",15,"bold")).pack(side=tk.LEFT, padx=20, pady=14)
tk.Label(header, text="MNIST · Decision Tree · Neural Network · Genetic Attack · Gaussian Defense",
         bg=PANEL_BG, fg=T_SEC, font=("Courier",8)).pack(side=tk.LEFT)
true_lbl = tk.Label(header, text="TRUE LABEL:  —", bg=PANEL_BG, fg=A_YLW,
                    font=("Courier",11,"bold"))
true_lbl.pack(side=tk.RIGHT, padx=20)
tk.Frame(window, bg=A_BLUE, height=2).pack(fill=tk.X)

# Body
body = tk.Frame(window, bg=BG)
body.pack(fill=tk.BOTH, expand=True, padx=14, pady=10)

# Sidebar
sidebar = tk.Frame(body, bg=PANEL_BG, width=175)
sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0,12))
sidebar.pack_propagate(False)

tk.Label(sidebar, text="CONTROLS", bg=PANEL_BG, fg=T_PRI,
         font=("Courier",11,"bold")).pack(pady=(18,4))
tk.Frame(sidebar, bg=BORDER, height=1).pack(fill=tk.X, padx=12)

tk.Label(sidebar, text="EPSILON  (attack strength)", bg=PANEL_BG, fg=T_SEC,
         font=("Courier",7,"bold")).pack(anchor="w", padx=14, pady=(14,2))
epsilon_var = tk.DoubleVar(value=0.1)
tk.Scale(sidebar, from_=0.05, to=0.25, resolution=0.05, orient=tk.HORIZONTAL,
         variable=epsilon_var, bg=PANEL_BG, fg=T_PRI, troughcolor=BORDER,
         highlightthickness=0, length=145, font=("Courier",7)).pack(padx=12)

tk.Label(sidebar, text="SIGMA  (defense strength)", bg=PANEL_BG, fg=T_SEC,
         font=("Courier",7,"bold")).pack(anchor="w", padx=14, pady=(14,2))
sigma_var = tk.DoubleVar(value=1.0)
tk.Scale(sidebar, from_=0.5, to=2.0, resolution=0.5, orient=tk.HORIZONTAL,
         variable=sigma_var, bg=PANEL_BG, fg=T_PRI, troughcolor=BORDER,
         highlightthickness=0, length=145, font=("Courier",7)).pack(padx=12)

tk.Label(sidebar, text="ANIM SPEED", bg=PANEL_BG, fg=T_SEC,
         font=("Courier",7,"bold")).pack(anchor="w", padx=14, pady=(14,2))
speed_var = tk.IntVar(value=80)
tk.Scale(sidebar, from_=20, to=300, resolution=20, orient=tk.HORIZONTAL,
         variable=speed_var, bg=PANEL_BG, fg=T_PRI, troughcolor=BORDER,
         highlightthickness=0, length=145, font=("Courier",7)).pack(padx=12)

tk.Frame(sidebar, bg=BORDER, height=1).pack(fill=tk.X, padx=12, pady=10)

for txt, cmd, col in [("LOAD IMAGE", load_image, A_BLUE),
                       ("RUN ATTACK", run_attack, A_RED),
                       ("APPLY DEFENSE", run_defense, A_GRN),
                       ("RESET", reset_all, T_DIM)]:
    tk.Button(sidebar, text=txt, command=cmd, bg=col, fg=BG,
              activebackground=col, relief=tk.FLAT, font=("Courier",9,"bold"),
              cursor="hand2", pady=7, width=16).pack(padx=12, pady=3)

tk.Frame(sidebar, bg=BORDER, height=1).pack(fill=tk.X, padx=12, pady=(14,6))
tk.Label(sidebar, text="NETWORK", bg=PANEL_BG, fg=T_SEC,
         font=("Courier",7,"bold")).pack()
nn_canvas = tk.Canvas(sidebar, width=160, height=160, bg=CARD_BG, highlightthickness=0)
nn_canvas.pack(padx=5, pady=4)
tk.Button(sidebar, text="REPLAY ANIM",
          command=lambda: animate_network(current["image"]) if current["image"] else None,
          bg=BORDER, fg=T_SEC, relief=tk.FLAT, font=("Courier",7), cursor="hand2").pack(pady=4)

# Three cards
content = tk.Frame(body, bg=BG)
content.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

def make_card(parent, title, accent):
    card = tk.Frame(parent, bg=CARD_BG)
    card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
    tk.Frame(card, bg=accent, height=3).pack(fill=tk.X)
    tk.Label(card, text=title, bg=CARD_BG, fg=accent,
             font=("Courier",10,"bold")).pack(pady=(10,4))
    img_lbl = tk.Label(card, bg="#0a0c10", width=160, height=160)
    img_lbl.pack(pady=6, padx=10)
    pred_lbl = tk.Label(card, text="—", bg=CARD_BG, fg=T_SEC,
                        font=("Courier",12,"bold"), justify=tk.LEFT)
    pred_lbl.pack(pady=10)
    return img_lbl, pred_lbl

orig_img, orig_pred_lbl = make_card(content, "ORIGINAL",      A_BLUE)
atk_img,  atk_pred_lbl  = make_card(content, "AFTER ATTACK",  A_RED)
def_img,  def_pred_lbl  = make_card(content, "AFTER DEFENSE", A_GRN)

# Status bar
tk.Frame(window, bg=BORDER, height=1).pack(fill=tk.X)
status_bar = tk.Frame(window, bg=PANEL_BG, height=30)
status_bar.pack(fill=tk.X)
status_bar.pack_propagate(False)
status_lbl = tk.Label(status_bar, text="Load an image to begin.",
                      bg=PANEL_BG, fg=T_SEC, font=("Courier",8), anchor="w")
status_lbl.pack(side=tk.LEFT, padx=16, pady=6)
tk.Label(status_bar, text="Group: Ayesha · Arwa · Eshaal",
         bg=PANEL_BG, fg=T_DIM, font=("Courier",7)).pack(side=tk.RIGHT, padx=16)

draw_network(nn_canvas)
window.mainloop()