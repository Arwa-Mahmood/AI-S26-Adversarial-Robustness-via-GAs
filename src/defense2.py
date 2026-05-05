# enhanced_defense.py
import numpy as np
from scipy.ndimage import median_filter, gaussian_filter
#from sklearn.ensemble import VotingClassifier

class EnhancedDefense:
    def __init__(self):
        self.defense_methods = {
            'gaussian': self._gaussian_defense,
            'median': self._median_defense,
            'ensemble': self._ensemble_defense,
            #'adaptive': self._adaptive_defense
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
        """Voting ensemble of multiple methods"""
        img_2d = image.reshape(28, 28)
        
        # Multiple defenses
        gaussian = gaussian_filter(img_2d, sigma=sigma)
        median = median_filter(img_2d, size=3) 
        
        # Weighted average (give more weight to median for outlier removal)
        #result = (gaussian * 0.4 + median * 0.6)
        result = (gaussian * 0.7 + median * 0.3)
        #return result.flatten()
        return np.clip(result, 0, 1).flatten()

    # def _adaptive_defense(self, image, base_sigma=1.0):
    #     """Adapt sigma based on local noise"""
    #     img_2d = image.reshape(28, 28)
        
    #     # Estimate noise per region
    #     smooth = gaussian_filter(img_2d, sigma=0.5)
    #     noise = np.abs(img_2d - smooth)
        
    #     # Higher sigma in noisy regions
    #     adaptive_sigma = base_sigma + (noise * 2)
    #     adaptive_sigma = np.clip(adaptive_sigma, 0.5, 3.0)
        
    #     # Apply adaptive smoothing
    #     from scipy.ndimage import gaussian_filter
    #     result = gaussian_filter(img_2d, sigma=adaptive_sigma)
    #     return result.flatten()
    
    def defend(self, X, method='ensemble', **kwargs):
        """Main defense interface"""
        defense_fn = self.defense_methods.get(method, self._ensemble_defense)
        
        if len(X.shape) == 1:
            return defense_fn(X, **kwargs)
        else:
            return np.array([defense_fn(x, **kwargs) for x in X])