import numpy as np
from scipy.ndimage import median_filter, gaussian_filter

class EnhancedDefense:
    def __init__(self):
        self.defense_methods = {
            'gaussian': self._gaussian_defense,
            'median': self._median_defense,
            'ensemble': self._ensemble_defense,
            'adaptive': self._adaptive_defense,
        }

    
    def _gaussian_defense(self, image, sigma=1.0):
        img_2d = image.reshape(28, 28)
        result = gaussian_filter(img_2d, sigma=sigma)
        return np.clip(result, 0, 1).flatten()
    

    def _median_defense(self, image, size=3):
        img_2d = image.reshape(28, 28)
        result = median_filter(img_2d, size=size)
        return np.clip(result, 0, 1).flatten()
    

    def _ensemble_defense(self, image, sigma=1.0):
        img_2d = image.reshape(28, 28)
        
        # removes salt-and-pepper adversarial perturbations while preserving digit edges
        result = median_filter(img_2d, size=3)
        
        # gentle gaussian to smooth remaining noise without blurring the digit
        effective_sigma = min(sigma * 0.35, 0.6)
        result = gaussian_filter(result, sigma=effective_sigma)
        
        return np.clip(result, 0, 1).flatten()


    def _adaptive_defense(self, image, sigma=1.0):
        img_2d = image.reshape(28, 28)
        
        # Estimate local noise by comparing with a lightly smoothed version
        light_smooth = gaussian_filter(img_2d, sigma=0.3)
        noise_map = np.abs(img_2d - light_smooth)
        noise_level = noise_map.mean()
        
        result = median_filter(img_2d, size=3)
        
        # Add Gaussian only if noise is significant
        if noise_level > 0.03:
            g_sigma = min(sigma * 0.3, 0.5)
            result = gaussian_filter(result, sigma=g_sigma)
        
        return np.clip(result, 0, 1).flatten()
    

    #main defense interface 
    def defend(self, X, method='ensemble', **kwargs):
        defense_fn = self.defense_methods.get(method, self._ensemble_defense)
        
        if len(X.shape) == 1:
            return defense_fn(X, **kwargs)
        else:
            return np.array([defense_fn(x, **kwargs) for x in X])