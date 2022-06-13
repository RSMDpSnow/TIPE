import ml.src.function as func
import numpy as np
import time

class Layer():
    def __init__(self, size, activ=func.none):
        self.size = size
        self.activ = activ

class Model():
    def __init__(self, name=""):
        self.L = 0
        self.name = name
        self.s = []
        self.e = []
        self.W = []
        self.B = []
        self.f = []
        self.n = []

    def __repr__(self):
        string = f"Model {self.name} :"
        for i in range(self.L):
            string += "\n|\t" + f"layer {i}: size={self.n[i]}; activ={self.f[i]};"
        return string
    
    def df(self, i):
        return func.Derivate(self.f[i])
    
    def add_layers(self, *layers):
        l0 = len(layers)
        for layer in layers:
            self.e.append( np.zeros((layer.size, 1)) )
            self.s.append( np.zeros((layer.size, 1)) )
            self.n.append( layer.size )
            self.f.append( layer.activ )
           

        for i in range(l0-1):
            l1, l2 = layers[i].size, layers[i+1].size
            self.W.append( np.random.random(size=l2*l1).reshape(l2, l1))  
        
        self.B.extend(np.random.random(l0)*0)
        self.L += l0
        
    def run(self, X):
        self.s[0] = X.copy()
        for i in range(self.L-1):
            self.e[i+1] = (self.W[i] @ self.s[i]) + self.B[i]
            self.s[i+1] = self.f[i](self.e[i+1])
            
    def predict(self, X):
        self.run(X)
        return self.s[-1]

    def train(self, X, Y, solver, epochs=10, debug=False):
        assert X.ndim==Y.ndim==2 and X.shape[1] == Y.shape[1]
        solver.init_data(X,Y)
        for epoch in range(epochs):
            if debug == True : t1 = time.perf_counter()
            
            J = solver.solve(self, X, Y)
            
            if debug == True:
                t2 = time.perf_counter(); deltaT = t2-t1
                print(f"epochs : {epoch+1} | processing time : {deltaT}, cost={J}")
                print()
        
        