class Value: 

    def __init__(self,data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda:None
        self._prev = set(_children)
        self._op = _op

    
    def __add__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self,other), '+')

        def _backward():
            # The gradient flows through the system in the addition process. You can see this from the Chain Rule.
            # Used '+=' because if is there any neuron replication then you also have to accumulate the gradients.
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out
    

    def __mul__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self,other), '*')

        def _backward():
            # In multiplication gradient of a node is just multiplications of its neighbours times the gradient of the next node. You can see this from the Chain Rule.
            # Used '+=' because if is there any neuron replication then you also have to accumulate the gradients.
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward    
        return out


    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out


    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            # Same logic as in multiplication but in addition we also have to use the Power Rule from derivative. You can see this from the Chain Rule.
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out
    

    def backward(self):
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # Go one variable at a time and apply the chain rule to get its gradient
        # Start from the last because derivative of something with respect to itself is 1 
        self.grad = 1
        for v in reversed(topo):
            # Python knows how v created with which operation and which Value instances so it will find the right backward func. Because we initiated ._backward = lambda:None and used __add__ and __mul__ functions from python with returning not value but a Value object.
            v._backward()




    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"


