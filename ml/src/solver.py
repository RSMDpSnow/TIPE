import ml.src.function as func
import numpy as np

class GradientDescenteMiniBatch():
    def __init__(self, cost=func.mse, learning_rate=0.1, batch_size=1):
        self.learning_rate = learning_rate
        self.cost = cost
        self.batch_size = batch_size
    
    def init_data(self, X, Y):
        self.training_data = [(X[:, k:k+self.batch_size],
                               Y[:, k:k+self.batch_size])
                              for k in range(X.shape[1])]
    def solve(self, model, X, Y):
        return 0

class GD:
    def __init__(self, cost=func.mse, learning_rate=0.1):
        self.cost = cost
        self.learning_rate = learning_rate
    
    def init_data(self, X, Y):
        self.nb_exp = X.shape[1]
        self.training_data = [(X[:,k], Y[:, k]) for k in range(self.nb_exp)]
    
    def solve(self, model, X, Y):
        dE_dW = [np.zeros(w.shape) for w in model.W]
        dE_db = [0]*self.nb_exp
        for (x,y) in self.training_data:
            delta = model.predict(x) - y
            dE_ds = [np.zeros(s.shape) for s in model.s]
            dE_ds[-1] = delta
            for l in reversed(range(model.L-1)):
                df_e = model.df(l)(model.e[l+1])
                for i in range(model.n[l]):
                    for j in range(model.n[l+1]):
                        dE_ds[l][i] += dE_ds[l+1][j] * df_e[j] * model.W[l][j,i] 
            
            for l in reversed(range(model.L-1)):
                df_e = model.df(l)(model.e[l+1])
                for i in range(dE_dW[l].shape[0]):
                    for j in range(dE_dW[l].shape[1]):
                        dE_dW[l][i,j] += model.s[l][j] * df_e[i] * dE_ds[l][j]
                dE_db[l] += np.dot(df_e, dE_ds[l+1])
                        
            
        for i in range(model.L - 1):
            model.W[i] -= dE_dW[i] * self.learning_rate / self.nb_exp
            #model.B[i] -= dE_db[i] * self.learning_rate / self.nb_exp
        
        return func.mse(model.predict(X)-Y)