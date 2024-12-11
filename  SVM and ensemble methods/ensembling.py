from base_ensemble import *
from utils import *
from sklearn.utils import resample
import traceback
import numpy as np
from collections import defaultdict

class TreeNode:

    __slots__ = ['depth', 'children', 'is_leaf', 'value', 'split_index', 'median', 'info_gain']
    def __init__(self, depth, info_gain = None, is_leaf = False, value = 0, split_index = None):

        #to split on column
        self.depth = depth

        #add children afterwards
        self.children = dict()

        #if leaf then also need value
        self.value = None
        self.info_gain = info_gain
        self.is_leaf = is_leaf
        self.median = 0
        if(self.is_leaf):
            self.value = value
        
        self.split_index = None
        if(not self.is_leaf):
            self.split_index = split_index

    def __repr__(self):
        '''
        Returns: string representation of the node
        '''
        return str(self)

    def __str__(self):

        return f'Depth = {self.depth} | IS_LEAF : {self.is_leaf} and value : {self.value} | SPLIT ON : {self.split_index} and GAIN : {self.info_gain}'

class DecisionTree:
    __slots__ = ['root', 'max_depth', 'feature_types', 'node_level_dict', 'prunes']
    
    def __init__(self, max_depth=10):
        '''
        Constructor
        '''
        self.root = None
        self.max_depth = max_depth
        self.node_level_dict = defaultdict(lambda : [])  # to store the nodes at each depth

    def entropy(self, data: np.ndarray, weights: np.ndarray):
        '''
        Calculates the entropy of the given data considering the sample weights.

        Args: data: numpy array of shape [num_examples, num_features + 1] : Last column is the target
              weights: numpy array of shape [num_samples] representing the sample weights.
        Returns: The weighted entropy of the data.
        '''
        _, counts = np.unique(data[:, -1], return_counts=True)
        probs = counts / data.shape[0]
        weighted_entropy = -np.sum(probs * np.log2(probs))

        # Apply the sample weights when calculating entropy
        weighted_entropy = np.sum(weights * weighted_entropy) / np.sum(weights)
        return weighted_entropy

    def information_gain(self, data: np.ndarray, attr_index: int, weights: np.ndarray):
        '''
        Calculates the information gain of the attribute at attr_index using the given data considering sample weights.

        Args: data: numpy array of shape [num_examples, num_features + 1] : Last column is the target
              attr_index: index of the attribute to split on and calculate the information gain
              weights: numpy array of shape [num_samples] representing the sample weights.
        '''
        mutual_info = self.entropy(data, weights)  # H(Y)

        if self.feature_types[attr_index] == 1:  # Categorical
            attr_vals, attr_counts = np.unique(data[:, attr_index], return_counts=True)
            attr_probs = attr_counts / data.shape[0]
            for i in range(len(attr_vals)):
                weighted_entropy = self.entropy(data[data[:, attr_index] == attr_vals[i]], weights)
                mutual_info -= attr_probs[i] * weighted_entropy
        else:  # Continuous
            median_val = np.median(data[:, attr_index])  # find median and split on median
            left = data[data[:, attr_index] <= median_val]
            right = data[data[:, attr_index] > median_val]

            weighted_entropy_left = self.entropy(left, weights)
            weighted_entropy_right = self.entropy(right, weights)
            mutual_info = mutual_info - ((len(left) * weighted_entropy_left + len(right) * weighted_entropy_right) / len(data))

        return mutual_info

    def build_tree(self, data: np.ndarray, features: list, depth: int, weights: np.ndarray):
        '''
        Recursively build the decision tree using the given data and features, and sample weights.

        Args: data: numpy array of shape [num_examples, num_features + 1] : Last column is the target
              features: list of indices of available attributes to split on
              depth: depth of the node in the tree
              weights: numpy array of shape [num_samples] representing the sample weights
        Returns: root of the decision tree
        '''
        val = 1 if np.sum(data[:, -1] == 1) > np.sum(data[:, -1] == -1) else -1
        if depth == self.max_depth or len(np.unique(data[:, -1])) == 1:  # last depth or pure leaf
            node = TreeNode(depth, is_leaf=True, value=val)
            self.node_level_dict[depth].append(node)
            return node
        
        best_attr = None
        best_gain = -float('inf')
    
        for attr_index in features:
            gain_local = self.information_gain(data, attr_index, weights)
            if gain_local > best_gain:
                best_gain = gain_local
                best_attr = attr_index

        node = TreeNode(depth=depth, is_leaf=False, split_index=best_attr)
        node.info_gain = best_gain
        node.value = val

        if self.feature_types[best_attr] == 1:  # Categorical
            attr_vals = np.unique(data[:, best_attr])
            for attr in attr_vals:
                node.children[attr] = self.build_tree(data[data[:, best_attr] == attr], features, depth + 1, weights)
        else:  # Continuous
            median_val = np.median(data[:, best_attr])
            node.median = median_val
            left = data[data[:, best_attr] <= median_val]
            right = data[data[:, best_attr] > median_val]

            if len(left):
                node.children[0] = self.build_tree(left, features, depth + 1, weights)
            if len(right):
                node.children[1] = self.build_tree(right, features, depth + 1, weights)

        self.node_level_dict[depth].append(node)
        return node

    def fit(self, X, y, types, max_depth, sample_weights=None):
        '''
        Fits a decision tree to X and y considering sample weights.

        Args: X: numpy array of data [num_samples, num_features]
              y: numpy array of labels [num_samples, 1]
              types: list of types of features (0: continuous, 1: categorical)
              max_depth: maximum depth of the tree
              sample_weights: optional numpy array of shape [num_samples], representing sample weights
        '''
        self.max_depth = max_depth
        self.feature_types = types
        y = y.reshape(-1, 1)
        #print(sample_weights)
        # Default to equal weights if no sample weights are provided
        if sample_weights is None:
            sample_weights = np.ones(X.shape[0]) / X.shape[0]
        
        self.root = self.build_tree(np.concatenate([X, y], axis=1), list(range(len(types))), 0, sample_weights)

    def recurive_predict(self, node, X):
        '''
        Recursively predicts the value for the given node and data

        Args: node: node of the tree
              X: numpy array of data [num_samples, num_features]
        Returns: predicted value
        '''
        if node.is_leaf:
            return node.value
        if self.feature_types[node.split_index] == 1:
            child_val = int(X[node.split_index])
            if child_val not in node.children:
                return node.value
            return self.recurive_predict(node.children[child_val], X)
        else:
            if X[node.split_index] <= node.median:
                if 0 not in node.children:
                    return node.value
                return self.recurive_predict(node.children[0], X)
            else:
                if 1 not in node.children:
                    return node.value
                return self.recurive_predict(node.children[1], X)

    def __call__(self, X):
        '''
        Predicted classes for X
        Args:
            X: numpy array of data [num_samples, num_features]
        Returns:
            y: [num_samples, 1] predicted classes
        '''
        return np.array([self.recurive_predict(self.root, x) for x in X])
    
class Decision_Tree:

    __slots__ = ['root', 'max_depth', 'feature_types', 'node_level_dict', 'prunes']
    def __init__(self,max_depth=10):
        '''
        Constructor
        '''
        self.root = None  
        self.max_depth = max_depth # max_Depth of the tree
        self.node_level_dict = defaultdict(lambda : []) # to store the nodes at each depth
    
    def entropy(self, data : np.ndarray):
        '''
        Calculates the entropy of the given data

        Args: data: numpy array of shape [num_example, num_features + 1] : Last column is the target
        Returns the entropy of the data
        '''
        _, counts = np.unique(data[:,-1], return_counts= True)
        probs = counts/data.shape[0] 
        return -np.sum(probs * np.log2(probs))

    def information_gain(self, data : np.ndarray, attr_index : int):
        '''
        Calculates the information gain of the attribute at attr_index using the given data

        Args: data: numpy array of shape [num_example, num_features + 1] : Last column is the target
              attr_index: index of the attribute to split on and calculate the information gain
        '''
        mutual_info = self.entropy(data) # H(Y)

        if self.feature_types[attr_index] == 1: # categorical

            attr_vals, attr_counts = np.unique(data[:,attr_index], return_counts = True)
            attr_probs = attr_counts / data.shape[0]
            for i in range(len(attr_vals)):
                mutual_info -= attr_probs[i] * self.entropy(data[data[:,attr_index] == attr_vals[i]])
        else : # continuous

            median_val = np.median(data[:,attr_index]) # find median and plit on median
            left = data[data[:,attr_index] <= median_val]
            right = data[data[:,attr_index] > median_val]
            mutual_info = mutual_info - ((len(left) * self.entropy(left)) + (len(right) * self.entropy(right)))/ len(data)

        return mutual_info

    def build_tree(self, data : np.ndarray, features : list, depth : int):
        '''
        Recursively build the decision tree using the given data and features

        Args:        data: numpy array of shape [num_example, num_features + 1] : Last column is the target
                 features: list of indices of available attributes to split on
                    depth: depth of the node in the tree
                
                  Returns: root of the decision tree'''

        val = 1 if np.sum(data[:, -1] == 1) > np.sum(data[:, -1] == -1) else -1
        #val = 1 if np.sum(data[:, -1] == 1) > np.sum(data[:, -1] == -1) else -1
        if depth == self.max_depth or len(np.unique(data[:,-1])) == 1: # last depth. predict the most common
            node = TreeNode(depth, is_leaf = True, value = val)
            self.node_level_dict[depth].append(node)
            #print(node.value)
            return node
        
        best_attr = None
        best_gain = -float('inf')
    
        for attr_index in features : #finding the best attribute with max informatoin gain
            gain_local = self.information_gain(data, attr_index)
            if gain_local > best_gain :
                best_gain = gain_local
                best_attr = attr_index

        node = TreeNode(depth=depth, is_leaf=False, split_index=best_attr)
        node.info_gain = best_gain
        node.value = val

        if self.feature_types[best_attr] == 1:
            attr_vals = np.unique(data[:,best_attr])
            for attr in attr_vals : #adding children corresponding to each attribute value
                node.children[attr] = self.build_tree( data[data[:, best_attr] == attr], features, depth +1)
        else :
            median_val = np.median((data[:,best_attr]))
            node.median = median_val
            left = data[ data[:,best_attr] <= median_val]
            right = data[ data[:,best_attr] > median_val]
            if len(left) :
                node.children[0] = self.build_tree(left, features, depth+1)
            if len(right):
                node.children[1] = self.build_tree(right, features, depth+1)

        self.node_level_dict[depth].append(node)
        #print(node.value)
        return node

    def fit(self, X, y, types ,max_depth):
        '''
        Fits a decision tree to X and y

        Args:       X: numpy array of data [num_samples, num_features]
                    y: numpy array of labels [num_samples, 1]
                types: list of types of features (0: continuous, 1: categorical)
                max_depth: maximum depth of the tree
        
        '''
        self.max_depth = max_depth
        self.feature_types = types
        y = y.reshape(-1, 1)
        self.root = self.build_tree(np.concatenate([X,y], axis = 1), list(range(len(types))), 0)

    def recurive_predict(self, node, X):
        '''
        Recursively predicts the value for the given node and data

        Args: node: node of the tree
                 X: numpy array of data [num_samples, num_features]

        Returns: predicted value
        '''
        
        if node.is_leaf:
            return node.value
        if self.feature_types[node.split_index] == 1:
            child_val = int(X[node.split_index])
            if child_val not in node.children:
                return node.value
            return self.recurive_predict(node.children[child_val], X)
        else :
            if X[node.split_index] <= node.median:
                if 0 not in node.children:
                    return node.value
                return self.recurive_predict(node.children[0], X)
            else :
                if 1 not in node.children:
                    return node.value
                return self.recurive_predict(node.children[1], X)

    def __call__(self, X):
        '''
        Predicted classes for X
        Args:
            X: numpy array of data [num_samples, num_features]
        Returns:
            y: [num_samples, 1] predicted classes
        '''
        return np.array([self.recurive_predict(self.root, x) for x in X])
    
    def post_prune(self, X_val, y_val, X_train, y_train, X_test, y_test):
        '''
        Post prunes the decision tree using the given validation data using a greedy approach
        Start trimming all nodes at the last level and move up the tree

        Args: X_val: numpy array of data [num_samples, num_features] for validation
                y_val: numpy array of labels [num_samples, 1] for validation
                X_train: numpy array of data [num_samples, num_features] for training
                y_train: numpy array of labels [num_samples, 1] for training
                X_test: numpy array of data [num_samples, num_features] for testing
                y_test: numpy array of labels [num_samples, 1] for testing

        Returns: val_acc_list: list of validation accuracies after each prune
                    train_acc: list of training accuracies after each prune
                     test_acc: list of testing accuracies after each prune
        '''
        val_best = np.mean(self(X_val) == y_val.T[0])
        val_acc_list = []; train_acc = []; test_acc = []
        print(f'Vaidation Accuracy = {val_best}')
        self.prunes = defaultdict(list)
        net_trimmed =0
        for level in range(self.max_depth - 1, -1, -1):
            node = self.node_level_dict[level]
            for n in node:
                if n.is_leaf == False:
                    n.is_leaf = True
                    val_acc = np.mean(self(X_val) == y_val.T[0])
                    if val_acc > val_best:
                        val_best = val_acc
                        val_acc_list.append(val_best)
                        train_acc.append(np.mean(self(X_train) == y_train.T[0]))
                        test_acc.append(np.mean(self(X_test) == y_test.T[0]))
                        self.prunes[level].append(n)
                        net_trimmed += 1
                    else: n.is_leaf = False
        print(f'Vaidation Accuracy = {val_best} | Total nodes trimmed = {net_trimmed}')
        return val_acc_list, train_acc, test_acc, net_trimmed



class RandomForestClassifier(BaseEnsembler):
    def __init__(self, num_trees=10, max_depth=2, feature_subsample_size=100):
        super().__init__(num_trees)
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.trees=[]

    def fit(self, X, y):
        '''
        Fit the RandomForest model by training multiple decision trees.
        
        Args:
            X : Input data (num_examples, num_features).
            y : Output labels (num_examples,).
        
        Output:
            None
        '''
        n_samples, n_features = X.shape
        base_seed = 42
        print(self.num_trees,self.max_depth,self.feature_subsample_size)
        for i in range(self.num_trees):
            # print(f"Training tree {i+1}/{self.num_trees}")
        
            # Set a unique seed for each tree by incrementing the base seed
            np.random.seed(base_seed + i)

            # Bootstrap sampling (sampling with replacement)
            try:
                X_bootstrap, y_bootstrap = resample(X, y, n_samples=n_samples, random_state=base_seed + i)
                # print(f"Bootstrap sampling completed for tree {i+1}")
            except Exception as e:
                print(f"Error during bootstrap sampling for tree {i+1}: {e}")
                traceback.print_exc()
                continue

            # Randomly select a subset of features
            try:
                all_features = list(range(n_features))
                if self.feature_subsample_size:
                    selected_features = np.random.choice(all_features, size=self.feature_subsample_size, replace=False)
                else:
                    selected_features = all_features  # Use all features if feature_subsample_size is not set
                    
                X_bootstrap = X_bootstrap[:, selected_features]  # Filter columns of X_bootstrap
                feature_types = [0] * len(selected_features)  # Assume all features are continuous for now
                # print(f"Feature subsampling completed for tree {i+1} with selected features {selected_features}")
            except Exception as e:
                print(f"Error during feature subsampling for tree {i+1}: {e}")
                traceback.print_exc()
                continue

            # Initialize and fit the decision tree with selected features
            tree = Decision_Tree(max_depth=self.max_depth)
            try:
                tree.fit(X_bootstrap, y_bootstrap, types=feature_types, max_depth=self.max_depth)
                # print(f"Tree {i+1} trained successfully")
            except Exception as e:
                print(f"Error during fit on tree {i+1}: {e}")
                traceback.print_exc()
                continue  # Continue training the next tree if there's an error

            self.trees.append(tree)

        print(f"Finished training RandomForest with {len(self.trees)} trees")


    def predict(self, X):
        '''
        Predict the class for each example in X using majority vote from all trees.
        
        Args:
            X : Input data (num_examples, num_features).
        
        Output:
            predictions : Predicted labels (num_examples,).
        '''
        predictions = np.zeros((self.num_trees, X.shape[0]))

        # Collect predictions from each tree
        for i, tree in enumerate(self.trees):
            predictions[i] = tree(X)
            #print(predictions[i])
        
        # Majority vote for each sample
        return np.array([np.sign(np.sum(p)) for p in predictions.T])  # Majority vote (sign of sum)

class AdaBoostClassifier(BaseEnsembler):
    def __init__(self, num_trees=100, max_depth=1, verbose=False):
        super().__init__(num_trees)
        self.alphas = []  # Weights for each weak classifier
        self.models = []  # List of trained models (weak classifiers)
        self.max_depth = max_depth
        self.verbose = verbose  # Whether to print debug information

    def fit(self, X, y):
        '''
        Fit the AdaBoost model by training weak classifiers iteratively and adjusting their weights.
        
        Args:
            X : Input data (num_examples, num_features).
            y : Output labels (num_examples,).
        
        Output:
            None
        '''
        n_samples = X.shape[0]
        weights = np.ones(n_samples) / n_samples  # Initialize equal weights for all samples
        self.models = []
        self.alphas = []

        for t in range(self.num_trees):
            # Train a weak classifier (decision tree) using the weighted data
            tree = DecisionTree()
            try:
                tree.fit(X, y, types=[0] * X.shape[1], max_depth=self.max_depth, sample_weights=weights)
            except Exception as e:
                print(f"Error during fit on tree {t + 1}: {e}")
                continue  # Skip this iteration if tree fitting fails

            try:
                predictions = tree(X)
            except Exception as e:
                print(f"Error during prediction on tree {t + 1}: {e}")
                continue  # Skip this iteration if prediction fails

            # Calculate the error rate and weight of the classifier
            incorrect = (predictions != y)
            error = np.sum(weights * incorrect) / np.sum(weights)
            if self.verbose:
                print(f"Tree {t + 1} - Weighted error: {error:.4f}")

            # Avoid division by zero or weak classifiers with too high error rate
            if error >= 0.4999999:
                #print(f"Stopping early: Error rate too high at tree {t + 1}")
                break

            # Calculate classifier weight (alpha)
            alpha = 0.5 * np.log((1 - error) / (error + 1e-10))  # Avoid division by zero
            self.alphas.append(alpha)
            self.models.append(tree)

            if self.verbose:
                print(f"Alpha for tree {t + 1}: {alpha:.4f}")

            # Update sample weights: more weight to misclassified samples
            weights *= np.exp(-alpha * y * predictions)
            weights /= np.sum(weights)  # Normalize weights

            if self.verbose:
                print(f"Updated weights after tree {t + 1}: {weights[:10]}")  # Print first 10 weights for debug

    def predict(self, X):
        '''
        Predict the class for each example in X by combining the weak classifiers.
        
        Args:
            X : Input data (num_examples, num_features).
        
        Output:
            predictions : Predicted labels (num_examples,).
        '''
        # Initialize the predictions array
        final_predictions = np.zeros(X.shape[0])

        # Sum the predictions of all weak classifiers weighted by their alpha
        for model, alpha in zip(self.models, self.alphas):
            final_predictions += alpha * model(X)

        # Return the final class predictions (sign of weighted sum)
        return np.sign(final_predictions)  # Return +1 or -1 for binary classification

# class AdaBoostClassifier(BaseEnsembler):
#     def __init__(self, num_trees=100,max_depth=1):
#         super().__init__(num_trees)
#         self.alphas = []  # Weights for each weak classifier
#         self.models = []  # List of trained models (weak classifiers)
#         self.max_depth=max_depth

#     def fit(self, X, y):
#         '''
#         Fit the AdaBoost model by training weak classifiers iteratively and adjusting their weights.
        
#         Args:
#             X : Input data (num_examples, num_features).
#             y : Output labels (num_examples,).
        
#         Output:
#             None
#         '''
#         n_samples = X.shape[0]
#         weights = np.ones(n_samples) / n_samples  # Initialize equal weights for all samples
#         self.models = []
#         self.alphas = []

#         for _ in range(self.num_trees):
#             # Train a weak classifier (decision tree) using the weighted data
#             tree = DecisionTree()
#             try:
#                 tree.fit(X, y, types=[0] * X.shape[1], max_depth=self.max_depth,sample_weights=weights)  # Shallow trees (stumps)
#             except Exception as e:
#                 print(f"Error during fit on tree")
#             try:
#                 predictions = tree(X)
#             except:
#                 print(f"Error during preduct on tree")

#             # Calculate the error rate and weight of the classifier
#             incorrect = (predictions != y)
#             error = np.sum(weights * incorrect) / np.sum(weights)
#             print(error)
#             # Avoid division by zero or weak classifiers with too high error rate
#             if error >= 0.5:
#                 break

#             # Calculate classifier weight (alpha)
#             alpha = 0.5 * np.log((1 - error) / (error + 1e-10))  # Avoid division by zero
#             self.alphas.append(alpha)
#             self.models.append(tree)
#             print(np.exp(-alpha * y * predictions))
#             # Update sample weights: more weight to misclassified samples
#             weights *= np.exp(-alpha * y * predictions)
#             weights /= np.sum(weights)  # Normalize weights
#             print(f"Tree predictions: {predictions[:10]}")
#             print(f"Weights after update: {weights[:10]}")

#     def predict(self, X):
#         '''
#         Predict the class for each example in X by combining the weak classifiers.
        
#         Args:
#             X : Input data (num_examples, num_features).
        
#         Output:
#             predictions : Predicted labels (num_examples,).
#         '''
#         # Initialize the predictions array
#         final_predictions = np.zeros(X.shape[0])

#         for model, alpha in zip(self.models, self.alphas):
#             final_predictions += alpha * model(X)

#         return np.sign(final_predictions)  # Return the final class predictions