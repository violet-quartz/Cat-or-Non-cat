import numpy as np

def sigmoid(z):
    """
    Arguments: z -- A scalar or numpy array.
    Return: sigmoid(z)
    """
    return 1 / ( 1 + np.exp(-z))

class LogisticRegression:

    def fit(self, X_train, Y_train, num_iter=1000, learning_rate=0.005, print_cost=False, cost_interval=100):
        """
        Train model.

        Arguments:
        X_train: data of size (dim, number of examples)
        Y_train: bool label vector of size (1, number of examples)

        Returns:
        params: weights w with size (dim, 1) and bias b which is a scalar 
        grads: gradients of weights and bias with respect to the cost function
        costs: list of all the costs computed during the optimization.
        """
        dim = X_train.shape[0]
        w, b = np.zeros((dim, 1)), 0.0

        costs = []
        for i in range(num_iter):
            grads, cost = self.propagate(w, b, X_train, Y_train)
            dw, db = grads['dw'], grads['db']
            w -= learning_rate * dw
            b -= learning_rate * db

            if i % cost_interval == 0:
                costs.append(cost)
                if print_cost:
                    print(f'Cost after iteration {i}: {cost}')
        
        params = {'w': w, 'b': b}
        grads = {'dw': dw, 'db': db}
        return params, grads, costs
    
    def predict(self, w, b, X, prob_threshold=0.5):
        """
        Arguments:
        w: model weights, ndarray of (dim, 1)
        b: bias, a scalar
        X: data of size (dim, number of examples)
        prob_threshold: probabilty threshold to predict 0 or 1

        Returns:
        Y_prediction: predictions (0/1) for the examples in X, ndarray of (1, number of examples)
        """
        m = X.shape[1]
        Y_prediction = np.zeros((1, m))
        A = sigmoid(np.dot(w.T, X) + b)
        for i in range(m):
            if A[0, i] > prob_threshold:
                Y_prediction[0, i] = 1
            else:
                Y_prediction[0, i] = 0
        return Y_prediction

    def propagate(self, w, b, X, Y):
        """
        Arguments:
        w: ndarray of (dim, 1)
        b: bias, a scalar
        X: ndarray of (dim, num of examples)
        Y: ndarray of (1, num of examples)

        Returns:
        grads: {dw, db}
        cost: negative log-likelihood cost for logistic regression
        """
        m = X.shape[1]
        A = sigmoid(np.dot(w.T, X) + b)
        cost = - np.mean(Y * np.log(A) + (1 - Y) * np.log(1 - A))
        dw = np.dot(X, (A - Y).T) / m
        db = np.mean(A - Y)
        grads = {'dw': dw, 'db':db }
        return grads, cost
    


