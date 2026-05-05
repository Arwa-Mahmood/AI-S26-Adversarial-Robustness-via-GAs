# defense.py - CORRECTED VERSION
import numpy as np
from sklearn.ensemble import IsolationForest

class EnhancedDefense:
    def __init__(self):
        self.detector = None  # Train on clean vs adversarial
        
    def _median_filter_defense(self, image, kernel_size=3):
        """Spatial filtering to remove outliers while preserving edges"""
        from scipy.ndimage import median_filter
        img_2d = image.reshape(28, 28)
        # Small kernel only - preserves details
        filtered = median_filter(img_2d, size=kernel_size)
        return filtered.flatten()
    
    def _feature_squeeze_defense(self, image, bit_depth=4):
        """Remove small perturbations by reducing precision"""
        squeezed = np.floor(image * (2**bit_depth)) / (2**bit_depth)
        return squeezed
    
    def _bit_depth_reduction(self, image, bits=3):
        """Extreme version - only 8 possible intensity levels"""
        levels = 2**bits
        quantized = np.floor(image * levels) / levels
        return quantized
    
    def defend(self, X, method='feature_squeeze', **kwargs):
        """Main defense interface"""
        if method == 'median':
            kernel = kwargs.get('kernel_size', 3)
            if len(X.shape) == 1:
                return self._median_filter_defense(X, kernel)
            return np.array([self._median_filter_defense(x, kernel) for x in X])
        
        elif method == 'feature_squeeze':
            bits = kwargs.get('bit_depth', 4)
            if len(X.shape) == 1:
                return self._feature_squeeze_defense(X, bits)
            return np.array([self._feature_squeeze_defense(x, bits) for x in X])
        
        elif method == 'bit_depth':
            bits = kwargs.get('bits', 3)
            if len(X.shape) == 1:
                return self._bit_depth_reduction(X, bits)
            return np.array([self._bit_depth_reduction(x, bits) for x in X])