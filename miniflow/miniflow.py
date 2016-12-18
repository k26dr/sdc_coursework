import numpy as np

class Layer:
    def __init__(self, inbound_layers=[]):
        self.inbound_layers = inbound_layers
        self.outbound_layers = []
        self.gradients = {}

        for n in self.inbound_layers:
            n.outbound_layers.append(self)
        self.value = None

    def forward(self):
        """
        Forward propagation. 

        Complete the output value based on inbound_layers and 
        store the result in self.value
        """
        raise NotImplemented

    def backward(self):
        "Backward Propagation"
        raise NotImplemented

class Input(Layer):
    def __init__(self):
        Layer.__init__(self)
    
    def forward(self, value=None):
        if value:
            self.value = value

    def backward(self):
        self.gradients = { self: 0 }
        for n in self.outbound_layers:
            self.gradients[self] += n.gradients[self] * 1

class Add(Layer):
    def __init__(self, *inputs):
        Layer.__init__(self, inputs)

    def forward(self):
        sum = 0
        for n in self.inbound_layers:
            sum += n.value
        self.value = sum
    
    def backward(self):
        self.gradients = {n: 0 for  n in self.inbound_layers}
        for n  in self.outbound_layers:
            for m in self.inbound_layers:
                self.gradients[m] += n.gradients[self] * 1


class Mult(Layer):
    def __init__(self, *inputs):
        Layer.__init__(self, inputs)
    
    def forward(self):
        product = 1
        for n in self.inbound_layers:
            product *= n.value
        self.value =  product

    def backward(self):
        self.gradients = {n: 0 for n in self.inbound_layers}
        product = reduce(lambda x,y: x*y.value, self.inbound_layers)
        for n  in self.outbound_layers:
            for m in self.inbound_layers:
                partial = product / m.value
                self.gradients[m] += n.gradients[self] * partial
    

class Linear(Layer):
    def __init__(self, inbound_layer, weights, bias):
        Layer.__init__(self, [inbound_layer, weights, bias])
        self.weights = weights
        self.bias = bias

    def forward(self):
        X = self.inbound_layers[0].value
        W = self.inbound_layers[1].value
        b = self.inbound_layers[2].value
        self.value = X.dot(W) + b

    def backward(self):
        self.gradients = {n: np.zeroes_like(n.value) for n in self.inbound_layers}
        X = self.inbound_layers[0].value
        W = self.inbound_layers[1].value
        b = self.inbound_layers[2].value

        for n in self.outbound_layers:
            grad_cost = n.gradients[self]
            self.gradients[W] = np.dot(grad_cost, X)
            self.gradients[X] = np.dot(grad_cost, W)
            self.gradients[b] = np.sum(grad_cost)
        

class Sigmoid(Layer):
    def __init__(self, layer):
        Layer.__init__(self, [layer])

    def _sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def forward(self):
        X = self.inbound_layers[0].value
        self.value = self._sigmoid(X)

class MSE(Layer):
    def __init__(self, y, a):
        Layer.__init__(self, [y, a])
    
    def forward(self):
         y = self.inbound_layers[0].value.reshape(-1, 1)
         a = self.inbound_layers[1].value.reshape(-1, 1)
         m = len(y)
         diff = y-a
         self.value = np.mean(diff**2)

def topological_sort(feed_dict):
    input_layers = [n for n in feed_dict.keys()]
    
    G = {}
    layers = [n for n in input_layers]
    while len(layers) > 0:
        n = layers.pop(0)
        if n not in G:
            G[n] = { 'in': set(), 'out': set() }   
        for m in n.outbound_layers:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            layers.append(m)
    
    L = []
    S = set(input_layers)
    while len(S) > 0:
        n = S.pop()
        
        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_layers:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            if len(G[m]['in']) == 0:
                S.add(m)
    return L

def forward_and_backward(graph):
    for n in graph:
        n.forward()

    for n in graph[::-1]:
        n.backward()
