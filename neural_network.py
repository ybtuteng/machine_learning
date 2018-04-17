import numpy as np
import layer
import math
import copy

HIDDEN_LAYER = 2
HIDDEN_DIM = 4
ALPHA = 0.1
LAMBDA = 0.1
OUTPUT_DIM = 1
ITERATION = 10000
INIT_EPSILON = 0.1
EPSILON = 0.0001


# hidden_dim:隐藏层神经元数目
# hidden_num:隐藏层层数
# input_deltas:对input值的偏导项，也就是神经元输入的误差，主要是为了
# weight:权重矩阵
# layers:神经网络层数组
# weight_deltas:权重偏导项矩阵


class NeuralNetwork:
    def __init__(self):
        self.weight = []
        self.layers = []
        self.weight_deltas = []

    # 初始化权重矩阵
    def init_weight(self, input_dim, hidden_dim, hidden_num):
        # input_dim + 1是因为输出层有一个偏置单元，下面同理
        self.weight.append((2 * INIT_EPSILON) * np.random.rand(input_dim + 1, hidden_dim) - INIT_EPSILON)
        for i in range(hidden_num - 1):
            self.weight.append((2 * INIT_EPSILON) * np.random.rand(hidden_dim + 1, hidden_dim) - INIT_EPSILON)
        self.weight.append((2 * INIT_EPSILON) * np.random.rand(hidden_dim + 1, OUTPUT_DIM) - INIT_EPSILON)

    # 基于反向传播算法的神经网络学习
    def back_propagation(self, training_data, hidden_dim, hidden_num):
        if hidden_num < 1:
            print("隐藏层数目不得小于1")
        input_dim = training_data[0][1].__len__()
        training_num = training_data.__len__()
        self.init_weight(input_dim, hidden_dim, hidden_num)
        print(self.weight)
        for i in range(hidden_num + 2):
            self.layers.append([])
        for times in range(ITERATION):
            print('第', times + 1, '次迭代')
            self.init_deltas(input_dim, hidden_dim, 1, hidden_num)
            cost = 0
            for y, x in training_data:
                # 正向传播，先构造输入层
                input_layer = layer.Layer(x)  # 将训练样本值放入输入层
                input_layer.set_output([1] + x)  # 输入层的输出值等于输入值，不必计算激活函数值，但需要增加一个偏差单元
                self.layers[0] = input_layer

                # 构造隐藏层和输出层
                for i in range(1, hidden_num + 2):
                    hidden_input = np.dot(self.layers[i - 1].get_output(), self.weight[i - 1])  # 根据上一层的输出计算本层输入值
                    hidden_layer = layer.Layer(hidden_input)  # 根据输入构造一个新的隐藏层
                    self.layers[i] = hidden_layer

                # 获得输出神经元的值
                output = self.layers[hidden_num + 1].get_output()  # 因为输出层里面包括了偏差单元，所以output[1]才是输出神经元
                cost += (y * math.log(output[1]) + (1 - y) * math.log(1 - output[1]))
                print('predict:', output[1], 'actual:', y)

                # 反向传播
                self.layers[hidden_num + 1].set_input_deltas([(output[1] - y) * output[1] * (1 - output[1])])  # 先算出输出层输入值的误差

                # 计算隐藏层和输入层的神经元输入值误差
                for i in range(hidden_num, -1, -1):
                    post_delta = self.layers[i + 1].get_input_deltas()
                    if i != hidden_num:
                        del post_delta[0]
                    input_deltas = self.layers[i].calculate_input_delta(self.weight[i], post_delta)
                    self.layers[i].set_input_deltas(input_deltas)

                # 计算隐藏层和输出层的权重误差，并累加到weight_deltas上
                for i in range(hidden_num + 1):
                    self.weight_deltas[i] = NeuralNetwork.calculate_weight_delta(self.weight_deltas[i],
                                                                                 self.layers[i].get_output(),
                                                                                 self.layers[i + 1].get_input_deltas())
            # 以所有样本权重误差累计值平均值作为偏导值，调整权重
            for l in range(hidden_num + 1):
                for i in range(self.weight[l].__len__()):
                    for j in range(self.weight[l][i].__len__()):
                        if i == 0:
                            self.weight_deltas[l][i][j] = self.weight_deltas[l][i][j] / training_num
                        else:
                            self.weight_deltas[l][i][j] = (self.weight_deltas[l][i][j] + LAMBDA * self.weight[l][i][
                                j]) / training_num
                        self.weight[l][i][j] = self.weight[l][i][j] - ALPHA * self.weight_deltas[l][i][j]

    # 初始化偏导矩阵为零矩阵
    def init_deltas(self, input_dim, hidden_dim, output_dim, hidden_num):
        # 初始化偏导矩阵
        self.weight_deltas = []
        self.weight_deltas.append(np.zeros([input_dim + 1, hidden_dim]))
        for i in range(hidden_num - 1):
            self.weight_deltas.append(np.zeros([hidden_dim + 1, hidden_dim]))
        self.weight_deltas.append(np.zeros([hidden_dim + 1, output_dim]))

    def gradient_checking(self, training_data, input_dim, hidden_dim, output_dim, hidden_num):
        deltas = list()
        deltas.append(np.zeros([input_dim + 1, hidden_dim]))
        for i in range(hidden_num - 1):
            deltas.append(np.zeros([hidden_dim + 1, hidden_dim]))
        deltas.append(np.zeros([hidden_dim + 1, output_dim]))
        for l in range(hidden_num + 1):
            for i in range(self.weight[l].__len__()):
                for j in range(self.weight[l][i].__len__()):
                    temp1 = self.calculate_cost(l, i, j, EPSILON, training_data)
                    temp2 = self.calculate_cost(l, i, j, -EPSILON, training_data)
                    deltas[l][i][j] = (temp1 - temp2) / 2 * EPSILON
        print(deltas)
        return deltas

    # 梯度检验
    def calculate_cost(self, l, i, j, epsilon, training_data):
        layers = []
        weight = copy.deepcopy(self.weight)
        hidden_num = weight.__len__() - 1
        weight[l][i][j] = weight[l][i][j] + epsilon
        cost = 0
        regularization = 0
        for i in range(hidden_num + 2):
            layers.append([])
        for y, x in training_data:
            # 正向传播
            input_layer = layer.Layer(x)
            input_layer.set_output([1] + x)  # 为训练样本值增加一个偏差单元
            layers[0] = input_layer

            for i in range(1, hidden_num + 1):
                hidden_input = np.dot(layers[i - 1].get_output(), weight[i - 1])
                hidden_layer = layer.Layer(hidden_input)
                layers[i] = hidden_layer

            output_input = np.dot(layers[hidden_num].get_output(), weight[hidden_num])
            output_layer = layer.Layer(output_input)
            layers[hidden_num + 1] = output_layer
            res = layers[hidden_num + 1].get_output()  # 因为res里面包括了偏差单元，所以res[1]才是输出
            cost += (y * math.log(res[1]) + (1 - y) * math.log(1 - res[1]))  # 按输出单元只有一个来举例

        for l in range(hidden_num + 1):
            for i in range(1, weight[l].__len__()):
                for j in range(weight[l][i].__len__()):
                    regularization += weight[l][i][j] * weight[l][i][j]
        cost = (-1 / training_data.__len__()) * cost + (LAMBDA / 2 * training_data.__len__()) * regularization
        return cost

    # 计算对权重的偏导数
    @staticmethod
    def calculate_weight_delta(weight_delta, output, post_deltas):
        weight_delta_matrix = np.array(weight_delta)
        output_matrix = np.array(output).reshape((1, output.__len__()))
        post_deltas_matrix = np.array(post_deltas).reshape((1, post_deltas.__len__()))
        res = weight_delta_matrix + output_matrix.T.dot(post_deltas_matrix)
        return res


if __name__ == '__main__':
    output_dim = 1
    hidden_dim = 5
    hidden_num = 1
    training_data = [(1, [1, 1, 1, 0, 0]), (1, [1, 0, 1, 0, 0]), (1, [0, 1, 0, 0, 0]), (0, [0, 0, 0, 1, 1]),
                     (0, [1, 0, 0, 0, 1]),
                     (0, [0, 0, 0, 1, 1]), (1, [1, 1, 1, 0, 0]), (1, [1, 0, 1, 0, 0]), (1, [0, 1, 0, 0, 0]),
                     (0, [0, 0, 0, 1, 1]), (0, [1, 0, 0, 0, 1]),
                     (0, [0, 0, 0, 1, 1]), (1, [1, 1, 1, 0, 0]), (1, [1, 0, 1, 0, 0]), (1, [0, 1, 0, 0, 0]),
                     (0, [0, 0, 0, 1, 1]), (0, [1, 0, 0, 0, 1]),
                     (0, [0, 0, 0, 1, 1]), (1, [1, 1, 1, 0, 0]), (1, [1, 0, 1, 0, 0]), (1, [0, 1, 0, 0, 0]),
                     (0, [0, 0, 0, 1, 1]), (0, [1, 0, 0, 0, 1]),
                     (0, [0, 0, 0, 1, 1]), (1, [1, 1, 1, 0, 0]), (1, [1, 0, 1, 0, 0]), (1, [0, 1, 0, 0, 0]),
                     (0, [0, 0, 0, 1, 1]), (0, [1, 0, 0, 0, 1]),
                     (0, [0, 0, 0, 1, 1]), (1, [1, 1, 1, 0, 0]), (1, [1, 0, 1, 0, 0]), (1, [0, 1, 0, 0, 0]),
                     (0, [0, 0, 0, 1, 1]), (0, [1, 0, 0, 0, 1]),
                     (0, [0, 0, 0, 1, 1])]
    network = NeuralNetwork()
    network.back_propagation(training_data, hidden_dim, hidden_num)
