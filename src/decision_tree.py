#DECISION TREE FROM SCRATCH 

def entropy(y): ...
    # -sum(p * log2(p)) for each class

def information_gain(X_col, y, threshold): ...
    # entropy(parent) - weighted_avg(entropy(left), entropy(right))

def best_split(X, y): ...
    # iterate all features + candidate thresholds, return best (feature, threshold)

import numpy as np
import pickle

#DECISION TREE FROM SCRATCH 

def entropy(y): ...
    # -sum(p * log2(p)) for each class

def information_gain(X_col, y, threshold): ...
    # entropy(parent) - weighted_avg(entropy(left), entropy(right))

def best_split(X, y): ...
    # iterate all features + candidate thresholds, return best (feature, threshold)


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, proba=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # Majority class
        self.proba = proba  # Class distribution for GA fitness


class DecisionTree:

    def __init__(self, max_depth=20, min_samples_split=2, n_features=None):
        self.max.depth = max_depth
        self.min_samples_split = min_samples_split #if a node has 2 images left then don't split further
        self.n_features = n_features 
        self.root = None 


    def fit(self, X, y): 
        # recursive build tree 
        self.n_classes = len(np.unique(y)) #seeing the number of classes
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)#checks all pixels
        self.root = self._built_tree(X,y) #building the tree


    def _build_tree(self, X, y, depth):

        n_samples, n_feats = X.shape 
        n_labels = len(np.unique(y))
    
        # Base cases: pure node, max_depth, min_samples
        if depth >= self.max_depth or n_labels == 1 or n_samples <self.min_samples_split:
            counts = np.bincount(y, minlength=self.n_classes)
            return Node(value=np.argmax(counts), proba=counts/n_samples)
        
        # Randomly picking a group of pixels to test
        feat_idxs = np.random.choice(n_feats,self.n_features, replace = False)

        #Which of these pixels, at what brightness level, splits the data most cleanly?
        best_feat, best_thr = self._best_split(X,y, feat_idxs)

        if best_feat is None: 
            counts = np.bincount(y, minlength=self.n_classes)
            return Node(valyue=np.argmax(counts), proba=counts/n_samples)
        
        left_idx = np.argwhere(X[:, best_feat] <=best_thr).flatten()
        right_idx =np.argwhere(X[:, best_feat] > best_thr).flatten()

        left=self._build_tree(X[left_idx, :], y[left_idx], depth+1)
        right=self._build_tree(X[right_idx, :], y[right_idx], depth+1)
                              
        return Node(feature=best_feat, threshold=best_thr, left=left_idx, right=right_idx)

    def predict(self, X):
        # traverse tree for each sample
        return np.array([self._traverse_tree(x, self.root, 'value') for x in X])

    def predict_proba(self, X): 
        # return class distribution at leaf (needed for GA fitness)
        return np.array([self._traverse_tree(x, self.root, 'proba') for x in X])


    def _traverse_tree(self, x, node, mode):
        if node.value is not None:
            return node.value if mode == 'value' else node.proba
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left, mode)
        return self._traverse_tree(x, node.right, mode)


    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.__dict__.update(data.__dict__)

