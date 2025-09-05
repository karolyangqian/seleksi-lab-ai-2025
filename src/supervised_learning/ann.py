import numpy as np

class Parameter:
    def __init__(self, data):
        self.data = data
        self.grad = np.zeros_like(data)

class Module:
    def __init__(self):
        self._parameters = {}
        self._submodules = {}

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._submodules[name] = value
        super().__setattr__(name, value)

    def zero_grad(self):
        for param in self.parameters():
            param.grad.fill(0)

    def parameters(self):
        for name, param in self._parameters.items():
            yield param
        for name, module in self._submodules.items():
            yield from module.parameters()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = Parameter(np.random.randn(out_features, in_features) * 0.01)
        self.bias = Parameter(np.zeros(out_features))
        self.input = None 
        
    def forward(self, x):
        self.input = x
        return np.dot(x, self.weight.data.T) + self.bias.data

    def backward(self, grad_output):
        self.weight.grad += np.dot(grad_output.T, self.input)
        self.bias.grad += np.sum(grad_output, axis=0)
        grad_input = np.dot(grad_output, self.weight.data)
        return grad_input

class ReLU(Module):
    def __init__(self):
        super().__init__()
        self.input = None

    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        relu_grad = (self.input > 0).astype(grad_output.dtype)
        return grad_output * relu_grad

class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, grad_output):
        sigmoid_grad = self.output * (1 - self.output)
        return grad_output * sigmoid_grad

class Softmax(Module):
    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        self.output = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        return self.output

    def backward(self, grad_output):
        return grad_output

class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = list(parameters)
        self.lr = lr

    def step(self):
        for param in self.parameters:
            param.data -= self.lr * param.grad

    def zero_grad(self):
        for param in self.parameters:
            param.grad.fill(0)
            
class SimpleNet(Module):
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.fc1 = Linear(input_size, hidden_size)
            self.relu = ReLU()
            self.fc2 = Linear(hidden_size, output_size)
            self.sigmoid = Sigmoid()

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return self.sigmoid(x)
        
        def backward(self, grad_output):
            grad = self.sigmoid.backward(grad_output)
            grad = self.fc2.backward(grad)
            grad = self.relu.backward(grad)
            grad = self.fc1.backward(grad)
            return grad

        def fit(self, x, y, epochs=1000, lr=0.01):
            if y.ndim == 1:
                y = y.reshape(-1, 1)

            optimizer = SGD(self.parameters(), lr=lr)
            for epoch in range(epochs):
                output = self.forward(x)
                
                loss = np.mean((output - y) ** 2)
                
                optimizer.zero_grad()
                
                grad_initial = 2 * (output - y) / y.shape[0]
                
                self.backward(grad_initial)
                
                optimizer.step()

                if epoch % 100 == 0:
                    print(f'Epoch {epoch}, Loss: {loss}')
        
        def predict(self, X):
            probabilities = self.forward(X)
            return (probabilities >= 0.5).astype(int)

        def predict_proba(self, X):
            prob_class_1 = self.forward(X)
            prob_class_0 = 1 - prob_class_1
            return np.hstack((prob_class_0, prob_class_1))