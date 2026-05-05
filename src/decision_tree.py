#DECISION TREE FROM SCRATCH 

import numpy as np
import pickle   #SAVE AND LOAD TRAINED MODEL TO/FROM DISK


#DECISION TREE FROM SCRATCH 

# def entropy(y): # y = array of class labels (digits 0 - 9)
#     if len(y) == 0: 
#         return 0.0 #entropy 0 -- no certainty 
    
#     # counting how many times each class appears 
#     counts = np.bincount(y, minlength = 10)

#     #counts to probabilities 
#     probs = counts / len(y)

#     #remove 0 probabilities  -- avoids log 0
#     probs = probs [probs > 0]

#     return -np.sum(probs * np.log2(probs))
#     # -sum(p * log2(p)) for each class -- entropy formula 

def gini(y):
    if len(y) == 0:
        return 0.0 
    
     # counting how many times each class appears 
    counts = np.bincount(y, minlength = 10)

    #counts to probabilities 
    probs = counts / len(y)

    #remove 0 probabilities  -- avoids log 0
    probs = probs [probs > 0]

    return 1 - np.sum(probs**2)


def information_gain(y, y_left, y_right):
    #measures how good a split is 
    if len(y_left) == 0 or len(y_right) == 0:
        return 0.0 
    
    #invalid split -- no gain 
    n = len(y)
    weighted = (len(y_left) / n) * gini (y_left) + (len(y_right) / n ) * gini(y_right) 

    #weighted avg after split  -- before -after
    return gini (y) - weighted 
    

class Node:

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, proba=None):
        self.feature = feature          # feature -- index of feature used for split 
        self.threshold = threshold      
        self.left = left                #left, right child nodes
        self.right = right
        self.value = value  # Majority class -- predicted class for leaf nodes
        self.proba = proba if proba is not None else [] # Class distribution for GA fitness -- probability distribution at leaf 


class DecisionTree:
    # MAX DEPTH , minimum samples required to split, N_FEATURES : no. of features to consider per split (random selection)
    def __init__(self, max_depth=20, min_samples_split=10, min_samples_leaf =5,  n_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split #if a node has 2 images left then don't split further
        self.n_features = n_features 
        self.root = None                        
        self.min_samples_leaf = min_samples_leaf       #---producing a splitting with enough children samples 


    # TRAINING 
    def fit(self, X, y): 
        # recursive build tree 
        self.n_classes = 10 # no. of distinct classes 
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)#checks all pixels 
        self.root = self._build_tree(X ,y, depth = 0) #building the tree


    def _build_tree(self, X, y, depth): 

        n_samples, n_feats = X.shape 
        n_labels = len(np.unique(y))
    
        # Base cases: pure node -- all samples same, max_depth, min_samples (too few)
        if depth >= self.max_depth or n_labels == 1 or n_samples <self.min_samples_split:
            counts = np.bincount(y, minlength=self.n_classes)
            return Node(value=int(np.argmax(counts)), proba=counts/n_samples)
        
        # Randomly picking a group of pixels to test
        feat_idxs = np.random.choice(n_feats,self.n_features, replace = False)

        #Which of these pixels, at what brightness level, splits the data most cleanly?
        best_feat, best_thr = self._best_split(X,y, feat_idxs)

        if best_feat is None: 
            counts = np.bincount(y, minlength=self.n_classes)
            return Node(value=int(np.argmax(counts)), proba=counts/n_samples)
        
        left_idx = np.argwhere(X[:, best_feat] <=best_thr).flatten()
        right_idx = np.argwhere(X[:, best_feat] > best_thr).flatten()

        left=self._build_tree(X[left_idx, :], y[left_idx], depth+1)
        right=self._build_tree(X[right_idx, :], y[right_idx], depth+1)
        
        return Node(feature=best_feat, threshold=best_thr, left=left, right=right)
    
    def _best_split(self, X, y, feat_idxs):
    # iterate all features + candidate thresholds, return best (feature, threshold)
        best_gain = -1          #intializing lwoest gain
        best_feat = None        
        best_thr = None 

        # looping over features 
        for feat in feat_idxs: 
            # extract feature column
            col = X[:, feat]

            #CANDIDATE THRESHOLDS -- percentiles instead of all values 
            thresholds = np.unique (np.percentile(col, [10,25,50,75,90]))

            # try each threshold 
            for thr in thresholds: 
                y_left = y[col <= thr]
                y_right = y[col > thr]

                if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                    continue                          # skip bad threshold, not the whole node

                #compute gain
                gain = information_gain(y, y_left, y_right)
                if gain > best_gain: 
                    best_gain = gain
                    best_feat = feat 
                    best_thr = thr 

        return best_feat, best_thr

    def predict(self, X):
        # traverse tree for each sample
        return np.array([self._traverse_tree(x, self.root, 'value') for x in X])

    def predict_proba(self, X): 
        # return class distribution at leaf (needed for GA fitness)
        return np.array([self._traverse_tree(x, self.root, 'proba') for x in X])


    def _traverse_tree(self, x, node, mode):
        #leaf node
        if node.value is not None:
            return node.value if mode == 'value' else node.proba
        
        #safety guard
        if node.left is None or node.right is None: 
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


#TREE -- testing

# if __name__ == "__main__":
#     import os
#     os.makedirs("models", exist_ok=True)

#     from mnist import load_mnist
#     X_train, y_train, X_test, y_test = load_mnist()

#     dt = DecisionTree(max_depth=20, min_samples_split=10, n_features=28)
#     dt.fit(X_train[:5000], y_train[:5000])

#     preds = dt.predict(X_test[:500])
#     print(f"Accuracy: {np.mean(preds == y_test[:500]):.4f}")

#     probas = dt.predict_proba(X_test[:5])
#     print(f"Proba shape: {probas.shape}")       # (5, 10)
#     print(f"Proba sums:  {probas.sum(axis=1)}") # all 1.0
if __name__ == "__main__":
    import os
    os.makedirs("models", exist_ok=True)

    X_train = np.load('x_train_final_data.npy')
    X_test  = np.load('x_test_final_data.npy')
    y_train = np.load('y_train_final_data.npy')
    y_test  = np.load('y_test_final_data.npy')

    dt = DecisionTree(max_depth=20, min_samples_split=10, n_features=28)
    dt.fit(X_train, y_train)

    preds = dt.predict(X_test)
    print(f"DT Accuracy: {np.mean(preds == y_test):.4f}")

    # dt.save('models/decision_tree.pkl')
    # print("Decision tree saved.")

    dt.save('models/decision_tree.pkl')
    print("Decision tree saved.")

    # made changes for eshaal