# __Author__: Hanseok Jo
import numpy as np
import random
import ActivationFunctions as af


class Layer(object):
    def __init__(self, typ, node_num, prev_layer=None):
        if prev_layer is None:
            return
        self.typ = typ
        self.node_num = node_num
        self.prev_layer = prev_layer
        self.input_number = self.prev_layer.output_vector
        self.input_vector = None
        self.output_vector = np.array([random.random() for y in range(self.node_num)])
        self.delta_vector = 0
        self.W = np.array([[random.uniform(0, 10) for x in range(len(self.input_number))] for y in range(self.node_num)])
        self.b = np.array([0 for y in range(self.node_num)])
        if prev_layer is None:
            return

    def set_input_vector(self):
        self.input_vector = np.dot(self.W, self.input_number) + self.b

    def set_output_vector(self):
        if self.typ == 'sigm':
            self.output_vector = np.vectorize(af.sigm)(self.input_vector)
        else:
            self.output_vector = np.vectorize(af.relu)(self.input_vector)

    def set_parameters(self):
        self.set_input_vector()
        self.set_output_vector()


class InputLayer(Layer):
    def __init__(self, input_number, typ=None, node_num=None):
        Layer.__init__(self, typ, node_num)
        self.input_vector = input_number
        self.output_vector = self.input_vector


class OutputLayer(Layer):
    def __init__(self, typ, target_vector, prev_layer):
        self.target_vector = target_vector
        self.node_num = len(self.target_vector)
        Layer.__init__(self, typ, self.node_num, prev_layer)


class NeuralNetwork(object):
    def __init__(self, input_data, target_data):
        self.input_vector = np.array(input_data)
        self.target_vector = np.array(target_data)

        self.input_layer = InputLayer(input_number=self.input_vector)
        self.hidden_layer = Layer(typ='sigm', node_num=4, prev_layer=self.input_layer)
        self.output_layer = OutputLayer(typ='sigm', target_vector=self.target_vector, prev_layer=self.hidden_layer)

    def fnnpropagate(self):
        self.hidden_layer.set_parameters()
        self.output_layer.set_parameters()
        pass

    def fnnbackpropagate(self, t_error=0.001, max_epoch=10000, learning_rate=0.01):
        def delta_update(output_layer, hidden_layer=None):
            update_layer = output_layer if hidden_layer is None else hidden_layer
            target_vector = output_layer.target_vector if hidden_layer is None else 0

            if update_layer.typ == 'sigm':
                func = np.vectorize(af.diff_sigm)
            else:
                func = np.vectorize(af.diff_relu)

            input_vector = update_layer.input_vector
            if update_layer == output_layer:
                output_vector = output_layer.output_vector
                output_layer.delta_vector = func(input_vector) * (target_vector - output_vector)
            else:
                output_vector = np.dot(output_layer.W.transpose(), output_layer.delta_vector)
                hidden_layer.delta_vector = func(input_vector) * output_vector

        def update_matrix(layer):
            layer.W += learning_rate * np.dot(layer.delta_vector, layer.input_vector)

        for epoch in range(max_epoch):
            self.fnnpropagate()
            error = 0
            for i in zip(self.target_vector, self.output_layer.output_vector):
                error += 0.5 * (i[0] - i[1]) ** 2
            print(error, epoch)
            if error < t_error:
                break

            delta_update(self.output_layer)
            delta_update(self.output_layer, self.hidden_layer)
            update_matrix(self.output_layer)
            update_matrix(self.hidden_layer)
            pass

input_numbers = np.array([1, 3, 4, 1])
target_numbers = np.array([2, 1, 10])

NN = NeuralNetwork(input_numbers, target_numbers)
NN.fnnpropagate()
NN.fnnbackpropagate(learning_rate=0.01)
