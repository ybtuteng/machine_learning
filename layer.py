import math
import numpy as np


class Layer:
    def __init__(self, input=[]):
        self.input = input
        self.output = []
        self.output = self.calculate_output(input)
        self.input_deltas = []  # 指每个神经元输入值的误差

    def calculate_output(self, input):
        self.output.clear()
        self.output.append(1)
        for neuron in input:
            self.output.append(self.squash(neuron))
        return self.output

    def calculate_input_delta(self, weight, post_delta):
        weight_matrix = np.array(weight)
        post_delta_matrix = np.array(post_delta)
        output_matrix = np.array(self.output)
        delta_matrix = weight_matrix.dot(post_delta_matrix) * output_matrix * (1 - output_matrix)
        self.input_deltas = delta_matrix.tolist()
        return self.input_deltas

    def set_output(self, output):
        self.output = output

    def set_input(self, input):
        self.input = input

    def set_input_deltas(self, delta):
        self.input_deltas = delta

    def get_output(self):
        return self.output

    def get_input(self):
        return self.input

    def get_input_deltas(self):
        return self.input_deltas

    # 激活函数sigmod
    def squash(self, input):
        return 1 / (1 + math.exp(-input))
