import random
from engine import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []


class Neuron(Module):
    
    def __init__(self,nin, nonlin=True):
        self.w =[Value(random.uniform(-1,1)) for _ in range (nin)]
        self.b = Value(0)
        self.nonlin = nonlin


    def __call__(self,x):
        # zip pairs every self.w with every corresponding x values.
        # we are just getting the cell body for the node (multiplying weights with inputs then sum every one of them and add bias.)
        act = sum((wi*xi for wi,xi in zip(self.w,x)) ,self.b)

        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


class Layer:

    def __init__(self,nin,nout, **kwargs):
        # Neuron(nin) for dimension
        # nout is for number of independent neurons for the layer.
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]
    
    def __call__(self,x):
        # Call the layer as array.
        # n(x) should call the neurons call func and do the math with the values that we give as params.
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"



class MLP(Module):

    def __init__(self, nin, nouts):
        # it will create an array of [ nin, nouts[0],...,nouts[n]]
        sz = [nin] + nouts
        # for example if we create a instance of MLP(3,[4,4,1]) that means we need a 3 layer (number of nouts) and first layers dimensionality should be 3 
        self.layers = [Layer(sz[i], sz[i+1],nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        # layer(x) should call the neurons call func and do the math with the values that we give as params.
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
    

#Neural Network
n = MLP(3,[4,4,1])
#Inputs
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
# Ground Truths (desired outcomes)
ys = [1.0, -1.0, -1.0, 1.0]


def predict(k):
    ##FORWARD PASS
    #Predictions
    ypred = [n(x) for x in xs]
    # sum of squares loss function 
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys,ypred))

    #BACKWARD PASS
    # Value class backward function
    #Clean the grads before the backpropagation.
    for p in n.parameters():
        p.grad = 0.0
    loss.backward()

    print(str(k)+"."+" PREDICTIONS: ")
    print(ypred)
    print(str(k)+"."+" LOSS: ")
    print(loss)
    return loss.data

def gradientDescent():
    step = 0.01
    # Gradient Descent
    for p in n.parameters():
        p.data += -step * p.grad

def train():
    k=1
    while True:
        loss = predict(k)
        gradientDescent()
        k +=1
        if loss < 0.0000001:
            break

train()