import unittest
from graphviz import Digraph
import numpy as np
import sys
sys.path.append('src')
from core.Autograd import Node, draw_dot, trace




class TestNode(unittest.TestCase):

    def test_add(self):
        a = Node(2)
        b = Node(3)
        c = a + b
        self.assertEqual(c.data, 5)

    def test_mul(self):
        a = Node(2)
        b = Node(3)
        c = a * b
        self.assertEqual(c.data, 6)

    def test_pow(self):
        a = Node(2)
        c = a ** 3
        self.assertEqual(c.data, 8)

    def test_tanh(self):
        a = Node(0)
        b = a.tanh()
        self.assertEqual(b.data, 0)

    def test_exp(self):
        a = Node(0)
        b = a.exp()
        self.assertEqual(b.data, 1)

    def test_relu(self):
        a = Node(-5)
        b = a.relu()
        self.assertEqual(b.data, 0)

    def test_div(self):
        a = Node(6)
        b = Node(3)
        c = a / b
        self.assertEqual(c.data, 2)

    def test_log(self):
        a = Node(np.e)
        b = a.log()
        self.assertEqual(b.data, 1)

    def test_sin(self):
        a = Node(np.pi/2)
        b = a.sin()
        self.assertEqual(b.data, 1)

    def test_cos(self):
        a = Node(0)
        b = a.cos()
        self.assertEqual(b.data, 1)

    def test_tan(self):
        a = Node(0)
        b = a.tan()
        self.assertEqual(b.data, 0)

    def test_backward(self):
        a = Node(2)
        b = Node(3)
        c = a * b
        c.backward()
        self.assertEqual(a.grad, 3)
        self.assertEqual(b.grad, 2)

    def test_visualize(self):
        a = Node(2)
        b = Node(3)
        c = a + b
        c.backward()
        dot = c.visualize()
        self.assertIsInstance(dot, Digraph)
        dot.view()


    def test_sub(self):
        a = Node(5)
        b = Node(3)
        c = a - b
        self.assertEqual(c.data, 2)

    def test_neg(self):
        a = Node(5)
        b = -a
        self.assertEqual(b.data, -5)

    def test_asin(self):
        a = Node(0.5)
        b = a.asin()
        self.assertAlmostEqual(b.data, np.arcsin(0.5))

    def test_acos(self):
        a = Node(0.5)
        b = a.acos()
        self.assertAlmostEqual(b.data, np.arccos(0.5))

    def test_atan(self):
        a = Node(1)
        b = a.atan()
        self.assertAlmostEqual(b.data, np.arctan(1))

    def test_log10(self):
        a = Node(10)
        b = a.log10()
        self.assertEqual(b.data, 1)

    def test_reset_gradients(self):
        a = Node(2)
        b = Node(3)
        c = a * b
        c.backward()
        a.reset_gradients()
        self.assertEqual(a.grad, 0)

    def test_trace(self):
        a = Node(2)
        b = Node(3)
        c = a + b
        nodes, edges = trace(c)
        self.assertIn(a, nodes)
        self.assertIn(b, nodes)
        self.assertIn(c, nodes)
        self.assertIn((a, c), edges)
        self.assertIn((b, c), edges)

      # Test Visualization
   

    # Add more tests for other functions...

if __name__ == '__main__':
    unittest.main()




