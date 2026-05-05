import numpy as np

def gaussian_kernel(size, sigma):
    ax = np.arange(-(size//2), size//2 + 1)                    # size=3, gives [-1, 0, 1]
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2)/(2*sigma**2))
    kernel = kernel/kernel.sum()
    return kernel

def apply_gaussian_smoothing(x, size=3, sigma=1.0):
    image = x.reshape(28, 28)                                   # flat image back to 28*28
    kernel = gaussian_kernel(size, sigma)                       # get the kernel
    output = np.zeros((28, 28))                                 # empty output image 
    pad = size//2                                               # pixels to pad
    padded=np.pad(image, pad, mode = 'constant', constant_values=0)
    for i in range(28):
        for j in range(28):
            region=padded[i:i+size, j:j+size]
            output[i,j] = np.sum(region*kernel)
    return output.flatten()

def defend(X, size=3, sigma=1.0):
    # return np.array([apply_gaussian_smoothing(x, size, sigma) for x in X])
    return np.array([apply_gaussian_smoothing(x, size, sigma) for x in X])

def evaluate_defense(X_clean, X_adv, y_true, predict_fn, sigmas=[0.5, 1.0, 1.5, 2.0]):
    for sigma in sigmas:
        x_clean_smoothed = defend(X_clean, size=3, sigma=sigma)
        clean_acc = np.mean(predict_fn(x_clean_smoothed) == y_true) * 100

        x_adv_smoothed = defend(X_adv, size=3, sigma=sigma)
        adv_acc = np.mean(predict_fn(x_adv_smoothed) == y_true) * 100

        print(f"Sigma {sigma:.1f} -> Clean Smoothed: {clean_acc:.2f}% | Adv Smoothed: {adv_acc:.2f}%")
