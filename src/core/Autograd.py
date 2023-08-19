import sys
sys.path.append('src')
import numpy as np
from Exception.Exception import DomainError, DivisionByZeroError, CircularReferenceError
from graphviz import Digraph

class Node:
    """A class representing a node in the computational graph."""

    def __init__(self, data, _children=(), _op="", label=""):
        # self.data = np.array(data)
        # self.grad = np.zeros_like(self.data)
        self.data = np.array(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data, dtype=np.float64)

        self._backward = lambda: None
        self._prev = set(_children) if _children else set()
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Node(data={self.data})"

    def visualize(self):
        """Visualize the computational graph starting from this node."""
        dot = draw_dot(self)
        return dot
    
    # Arithmetic operations
    def __add__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += np.ones_like(self.data) * out.grad
            other.grad += np.ones_like(other.data) * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        # assert isinstance(other, (int, float)), "Only supporting int/float powers for now"
        out = Node(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    # def __truediv__(self, other):
    #     return self * other**-1

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    # Activation functions
    def tanh(self):
        t = (np.exp(2 * self.data) - 1) / (np.exp(2 * self.data) + 1)
        out = Node(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Node(np.exp(self.data), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Node(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out
    
    def __truediv__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        if other.data == 0:
            raise DivisionByZeroError("Division by zero is not allowed.")
        out = Node(self.data / other.data, (self, other), '/')

        def _backward():
            self.grad += 1.0 / other.data * out.grad
            other.grad += -self.data / (other.data ** 2) * out.grad
        out._backward = _backward
        return out
    #  # Direct Division
    # def __div__(self, other):
    #     other = other if isinstance(other, Node) else Node(other)
    #     if other.data == 0:
    #         raise DivisionByZeroError("Division by zero is not allowed.")
    #     out = Node(self.data / other.data, (self, other), '/')

    #     def _backward():
    #         self.grad += 1.0 / other.data * out.grad
    #         other.grad += -self.data / (other.data ** 2) * out.grad
    #     out._backward = _backward
    #     return out

    # Logarithm (natural)
    def log(self):
        if self.data <= 0:
            print("Warning: Logarithm of non-positive value. Setting result to negative infinity.")
            out_data = float('-inf')
        else:
            out_data = np.log(self.data)
        out = Node(out_data, (self,), 'log')

        def _backward():
            if self.data > 0:
                self.grad += 1.0 / self.data * out.grad
        out._backward = _backward
        return out

    # Logarithm (base 10)
    def log10(self):
        if self.data <= 0:
            print("Warning: Logarithm of non-positive value. Setting result to negative infinity.")
            out_data = float('-inf')
        else:
            out_data = np.log10(self.data)
        out = Node(out_data, (self,), 'log10')

        def _backward():
            if self.data > 0:
                self.grad += 1.0 / (self.data * np.log(10)) * out.grad
        out._backward = _backward
        return out

    # Trigonometric Functions
    def sin(self):
        out = Node(np.sin(self.data), (self,), 'sin')

        def _backward():
            self.grad += np.cos(self.data) * out.grad
        out._backward = _backward
        return out

    def cos(self):
        out = Node(np.cos(self.data), (self,), 'cos')

        def _backward():
            self.grad += -np.sin(self.data) * out.grad
        out._backward = _backward
        return out

    def tan(self):
        out = Node(np.tan(self.data), (self,), 'tan')

        def _backward():
            self.grad += (1 + out.data ** 2) * out.grad
        out._backward = _backward
        return out

    # Inverse Trigonometric Functions
    def asin(self):
        if not (-1 <= self.data <= 1):
            raise DomainError("Domain error for arcsin.")
        out = Node(np.arcsin(self.data), (self,), 'asin')

        def _backward():
            self.grad += 1.0 / np.sqrt(1 - self.data ** 2) * out.grad
        out._backward = _backward
        
        return out

    def acos(self):
        if -1 <= self.data <= 1:
            out_data = np.arccos(self.data)
        else:
            print("Warning: Domain error for arccos. Setting result to NaN.")
            out_data = float('nan')
        out = Node(out_data, (self,), 'acos')

        def _backward():
            if -1 < self.data < 1:
                self.grad += -1.0 / np.sqrt(1 - self.data ** 2) * out.grad
        out._backward = _backward
        return out

    def atan(self):
        out = Node(np.arctan(self.data), (self,), 'atan')

        def _backward():
            self.grad += 1.0 / (1 + self.data ** 2) * out.grad
        out._backward = _backward
        return out
    
    def reset_gradients(self):
        """Reset gradients to zero."""
        self.grad = np.zeros_like(self.data)
        for child in self._prev:
            child.reset_gradients()


    

    def backward(self):
        topo = []
        visited = set()
        seen = set()

    

        def build_topo(v):
            if v in seen:
                raise CircularReferenceError("Circular reference detected in the computational graph.")
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
                visited.add(v)
                if v in seen:  # Add this check
                    seen.remove(v)
        build_topo(self)

        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()



def trace(root):
    #builds a set of all nodes and edges in a graph
    nodes, edges = set() , set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format="png", graph_attr={'rankdir': 'LR'})  # LR = left to right

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        dot.node(name=uid, label="{%s | data %.4f | grad %.4f }" % (n.label, np.mean(n.data), np.mean(n.grad)), shape='record')
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot

# x = Node([2, 3])
# y = Node([3, 4])
# z = x * y
# z.backward()
# z.visualize().view()