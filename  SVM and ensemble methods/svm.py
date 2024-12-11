from utils import *
import numpy as np
from cvxopt import matrix, solvers
from tqdm import tqdm  # For progress bars
from sklearn.model_selection import train_test_split

def getSupport(arr, tol=1e-3, C=1):
    supportAlpha=arr>tol
    return supportAlpha

def linearKernel(X1, X2):
    return np.matmul(X1, X2.T)

def gaussKernel(X1, X2, gamma=0.001):
    prod = np.reshape(np.einsum('ij,ij->i', X1, X1), (X1.shape[0], 1)) + \
           np.reshape(np.einsum('ij,ij->i', X2, X2), (X2.shape[0], 1)).T - 2 * np.matmul(X1, X2.T)
    return np.exp(-gamma * prod)


class SoftMarginSVMQP:
    def __init__(self, C=1.0, kernel='linear', gamma=1.0, tol=1e-4):
        '''
        Initializes the SVM with the specified kernel and regularization parameter.
        
        Args:
            C: Regularization parameter.
            kernel: Type of kernel ('linear' or 'rbf').
            gamma: Hyperparameter for the RBF kernel.
            tol: Tolerance level for support vector selection.
        '''
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.tol = tol
        self.b=None
        self.w=None
        self.alphas = None
        self.sv_X = None
        self.sv_y = None

    def _kernel(self, X1, X2):
        '''Computes the kernel function between two sets of vectors X1 and X2.'''
        if self.kernel == 'linear':
            return linearKernel(X1, X2)
        elif self.kernel == 'rbf':
            return gaussKernel(X1, X2, self.gamma)
        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel}")

    def fit(self, X, y):
        '''
        Trains the SVM model using quadratic programming to solve the dual problem.
        
        Args:
            X: Input data.
            y: Target labels.
        '''
        # print("Random Sampling")
        np.random.seed(42)  # Set the seed for reproducibility
        X, _, y, _ = train_test_split(X, y, train_size=7000, stratify=y, random_state=42)
        # Randomly choose indices from the data
        # print(len(X))
        # indices = np.random.choice(len(X), size=5000, replace=False)
        
        # # Select the samples from X and y using the chosen indices
        # X = X[:3000]
        # y = y[:3000]
        m, n = X.shape
        y = y.reshape(-1, 1) * 1.0  # Ensure y is a column vector
        # Kernel matrix with progress bar
        print("Calculating Kernel Matrix...")
        K = self._kernel(X, X)

        # Construct the QP parameters
        P = matrix(np.outer(y,y)*K)
        q = matrix(-np.ones(m))
        G = matrix(np.concatenate([np.eye(m), -np.eye(m)]))
        h = matrix(np.concatenate((self.C * np.ones(m), np.zeros(m))))
        A = matrix(y.reshape(1,-1))
        b = matrix(np.zeros(1))
        #self.print_matrices(P, q, G, h, A, b)
        # Solve QP problem
        print("Solving Quadratic Programming problem...")
        solvers.options['show_progress'] = True
        #solvers.options['maxiters'] = 10
        solution = solvers.qp(P, q, G, h, A, b)
        alphas = np.ravel(solution['x'])

        # Select support vectors using getSupport
        support_indices = alphas>self.tol
        self.alphas = alphas[support_indices]
        self.sv_X = X[support_indices]
        self.sv_y = y[support_indices]

        # Calculate intercept (b)
        # Calculate intercept (b)
        print("Calculating intercept (b) and (w)...")
            # Calculate weight vector `w` for linear kernel
        if self.kernel == 'linear':
            # Sum over all support vectors to get `w`
            self.w = np.sum((self.alphas.reshape(-1, 1) * self.sv_y.reshape(-1, 1)) * self.sv_X, axis=0)
            wXt = np.dot(X, self.w)
            try:
                positive_sv_indices = (y.flatten() == 1) & (self.C - alphas > self.tol)
                negative_sv_indices = (y.flatten() == -1) & (self.C - alphas > self.tol)

                # Extract maximum and minimum for the intercept calculation
                M = np.max(wXt[positive_sv_indices])
                m = np.min(wXt[negative_sv_indices])

                # Calculate intercept as the midpoint
                self.b = -(M + m) / 2
            except Exception as e:
                print(f"Error calculating : {e}")
            
        else:
            # For non-linear kernels, calculate `b` using support vectors
            try:
                K = self._kernel(self.sv_X, self.sv_X)
                self.b = np.mean(self.sv_y - np.dot((self.alphas * self.sv_y).T, K))
            except Exception as e:
                print(f"Error calculating : {e}")
    

    def predict(self, X):
        '''
        Predicts labels for input data.
        
        Args:
            X: Input data (shape: [n_samples, n_features]).
            
        Returns:
            Predicted labels (shape: [n_samples, ]).
        '''
        try:
            if self.kernel == 'linear':
                # Linear kernel prediction: w.X + b
                if X.shape[1] != self.w.shape[0]:
                    raise ValueError(f"Feature mismatch: X has {X.shape[1]} features, but the model expects {self.w.shape[0]} features.")
                pred = np.sign(np.dot(X, self.w) + self.b)

            elif self.kernel == 'rbf':
                # RBF kernel prediction
                K = self._kernel(X, self.sv_X)
                self.alphas = self.alphas.flatten()  # Make sure it's 1D
                self.sv_y = self.sv_y.flatten()
                # Compute the decision function: sum(alpha_i * y_i * K(X, X_i)) + b
                decision = np.dot(K, (self.alphas * self.sv_y)) + self.b
                pred = np.sign(decision)
                # Return the sign of the decision function (class predictions)
                return np.sign(decision)  # This will return a 1D vector

            else:
                raise ValueError(f"Unsupported kernel: {self.kernel}")

        except Exception as e:
            print(f"Error in prediction: {e}")
            pred = None

        return pred
