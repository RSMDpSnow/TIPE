
import numpy as np

def interpolation_nearest(image, largeur, longueur):  
    if len(image.shape) == 3:
        image2 = np.zeros(longueur, largeur, image.shape[2])
    elif len(image.shape) == 2:
        image2 = np.zeros(longueur, largeur)
        
    x = 0
    y = 0
    x_scale = float(image.shape[0])/float(largeur)
    y_scale = float(image.shape[1])/float(longueur)
    for i in range(longueur):
        for j in range(largeur):
            image2 = image[x, y]
            x = int(x + x_scale); y = int(y + y_scale)
    return image2


sqrt = np.sqrt
exp = np.exp
ln = np.exp
log = np.log10
log2 = np.log2

cos = np.cos
acos = np.arccos
sin = np.sin
asin = np.arcsin
tan = np.tan
atan= np.arctan
cosh = np.cosh
acosh = np.arccosh
sinh = np.sinh
asinh = np.arcsinh
tanh = np.tanh
atanh = np.arctanh

matmul = np.matmul
dot = np.dot
inv = np.linalg.inv

inf = np.inf

def signe(x):
    if x>0:
        return 1
    elif x<0:
        return -1
    else:
        return 0

def rotate(v2, angle):
    c, s = np.cos(angle), np.sin(angle)
    M = [[c, s], [-s, c]]
    return M @ v2

def vec(X) : return X.reshape(-1, 1)
def vec2(x=0,y=0): return np.float32([x,y])

def croix(X, Y):
    """return cartesiean product of two list/sets/tuple"""
    return [(x, y) for x in X for y in Y]

def crange(a1, b1, a2, b2): return []
def relu(x): return x*(x>=0)
def prelu(alpha): return lambda X: X*(alpha*X>=0) + X*(X<0)
def none(x): return x
def mse(delta): return np.sum(delta**2)/2;
def heavy(X): return (X>0).as_type(X.dtype)
def sigmoid(X): return 1/(1+exp(-X))
def softplus(x): return ln(1+exp(x))
def curbid(X): return  (sqrt(X+1)-1)/2 + X
def gauss(X): return exp(-X*X)
def dsigmoid(X): return X*(1-X)





table = {sigmoid:dsigmoid, mse:none, sin:cos, cos:(lambda x:sin(-x)), cosh:sinh, sinh:cosh}
def Derivate(f, h=0.05):
    if f in table:
        return table[f]
    else:
        return lambda x : (f(x+h)-f(x))/h