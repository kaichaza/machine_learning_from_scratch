# This script creates a callable class which runs a single Perceptron
# The perceptron is able to solve the logical OR, the logical AND, but not
# The logical XOR problem. The only library used is numpy.
#
# Code from Stephen Marsland, Machine Learning An Algorithmic Perspective, 2nd edition
# https://seat.massey.ac.nz/personal/s.r.marsland/Code/Ch3/pcn.py
# Chapter 3, basic perceptron


import numpy as np


class Perceptron:
    """This class implements a basic bare-bones neural perceptron

    Attributes:
        inputs: input vector
        target: target vector
        n_in: number of input features
        n_out: number of output classes
        weights: input weights
    Methods:
        train_step: trains the perceptron
        train_step: trains the perceptron
        train_serial_linear: trains the perceptron 1 datapoint at a time using
                           a linear Delta rule activation function
        fwd: run a forward learning algorithm
        confusion_matrix: calculate the confusion matrix
    """

    def __init__(self, inputs, targets):
        # Set up the network size
        if np.ndim(inputs) > 1:
            self.n_in = np.shape(inputs)[1]
        else:
            self.n_in = 1

        if np.ndim(targets) > 1:
            self.n_out = np.shape(targets)[1]
        else:
            self.n_out = 1

        self.n_data = np.shape(inputs)[0]
        self.activations = []

        # Now initialize the network
        self.weights = np.random.rand(self.n_in + 1, self.n_out) * 0.1 - 0.5

    def train_step(self, inputs, targets, learning_rate, num_iterations, verbose=False):
        """

        Train with a step function in the perceptron

        :param inputs: input vector
        :param targets: target vector
        :param learning_rate: this is "eta" given as a fraction
        :param num_iterations: to convergence
        :return: trained weights of the inputs based on the gradient
        """
        # Add the inputs that match the bias node
        # original code
        # inputs = np.concatenate((inputs, -np.ones((self.n_data, 1))), axis=1)
        # which I change to the below with the reversal of the bias sign
        inputs = np.concatenate((inputs, np.ones((self.n_data, 1))), axis=1)

        if verbose:
            print("Initial inputs: ", inputs)

        # Training
        change = range(self.n_data)
        self.errors_step = []

        for n in range(num_iterations):
            self.activations = self.fwd(inputs)
            self.weights -= learning_rate * np.dot(np.transpose(inputs), self.activations - targets)
            current_error = abs((self.activations - targets).sum())
            # randomnize the order of inputs
            # np.random.shuffle(change)
            # inputs = inputs[change, :]
            # targets = targets[change, :]
            self.errors_step.append(current_error)
            if verbose:
                print("iteration number: ", n)
                print("Weights")
                print(self.weights)
                print("Total error (number of incorrect predictions)")
                print(self.errors_step)
        return self.weights

    def train_linear(self, inputs, targets, learning_rate, num_iterations, verbose=False):
        """

        Train using the Delta-rule, using a linear function in the perceptron

        :param inputs: input vector
        :param targets: target vector
        :param learning_rate: this is "eta" given as a fraction
        :param num_iterations: to convergence
        :return: trained weights of the inputs based on the gradient
        """
        # Add the inputs that match the bias node
        # original code
        # inputs = np.concatenate((inputs, -np.ones((self.n_data, 1))), axis=1)
        # which I change to the below with the reversal of the bias sign
        inputs = np.concatenate((inputs, np.ones((self.n_data, 1))), axis=1)

        if verbose:
            print("Initial inputs: ", inputs)

        # Training
        self.cost_function = []

        for n in range(num_iterations):
            self.activations = self.fwd(inputs)

            errors = self.activations - targets
            self.weights -= (learning_rate * inputs.T.dot(errors))

            cost = (errors ** 2).sum() / 2.0
            self.cost_function.append(cost)

            # randomnize the order of inputs
            # np.random.shuffle(change)
            # inputs = inputs[change, :]
            # targets = targets[change, :]
            if verbose:
                print("iteration number: ", n)
                print("Weights")
                print(self.weights)
                print("Cost function, appended after each iteration")
                print(self.cost_function)
                print()
        return self.weights

    def fwd_serial(self, single_input_data_row, verbose):
        """Run the network forward"""
        # Compute output activations
        output = np.dot(single_input_data_row.T, self.weights)
        if verbose:
            print("double checking shape of fwd_serial calculation: (output) ")
            print(np.shape(np.dot(single_input_data_row.T, self.weights)))
            print("actual output: ", output)

        single_activation = 0
        if output == 0:
            single_activation = 0
        if output > 0:
            single_activation = 1

        return single_activation

    def train_serial_linear(self, inputs, targets, learning_rate, num_iterations, verbose=False):
        """

        Train using the Delta-rule, calculating each input point one at at time
        using a linear function in the perceptron

        :param inputs: input vector
        :param targets: target vector
        :param learning_rate: this is "eta" given as a fraction
        :param num_iterations: to convergence
        :return: trained weights of the inputs based on the gradient
        """
        # Add the inputs that match the bias node
        # original code
        # inputs = np.concatenate((inputs, -np.ones((self.n_data, 1))), axis=1)
        # which I change to the below with the reversal of the bias sign
        inputs = np.concatenate((inputs, np.ones((self.n_data, 1))), axis=1)

        if verbose:
            print("Initial inputs: ", inputs)

        # Training
        self.cost_function = []

        self.activations = np.zeros((np.shape(inputs)[0], np.shape(inputs)[1]))

        data_points = np.shape(inputs)[0]
        num_weights = np.shape(self.weights)[0]

        if verbose:
            print("shape of activations matrix: ", np.shape(self.activations))
            print("shape of weights matrix: ", np.shape(self.weights))
            print("shape of inputs matrix: ", np.shape(inputs))
            print("shape of targets matrix: ", np.shape(targets))
            print("data_points ", data_points)
            print("num_weights", num_weights)
            print("self.n_in + 1: ", self.n_in + 1)

        # Loop over the input vector here
        for data in range(data_points):
                single_input_data_row = np.zeros((self.n_in + 1, 1))
                single_input_data_row[:, 0] = inputs[data].T

                if verbose:
                    print("Iterating through the nth data point: ", data)
                    print("single input data row: ", single_input_data_row)
                    print("shape of input data row: ", np.shape(single_input_data_row))
                    print("shape of targets[data] ", np.shape(targets[data]))

                self.activations[data] = self.fwd_serial(single_input_data_row, verbose)

                error = np.zeros((3, 1))
                error[:, 0] = self.activations[data] - targets[data]

                if verbose:
                    print("error: ", error)
                    print("shape of activations error: ", np.shape(error))

                self.weights -= (learning_rate * np.multiply(inputs[data].T, error.T)).T

                cost = (error ** 2).sum() / 2.0
                self.cost_function.append(cost)

        if verbose:
            print("Weights")
            print(self.weights)
            print("Cost function, appended after each iteration")
            print(self.cost_function)
            print()

        return self.weights

    def fwd(self, inputs):
        """Run the network forward"""
        # Compute activations
        self.activations = np.dot(inputs, self.weights)

        # Threshold the activations
        return np.where(self.activations > 0, 1, 0)

    def confusion_matrix(self, con_inputs, con_targets, verbose=False):
        # Add the inputs that match the bias node

        con_inputs = np.concatenate((con_inputs, -np.ones((self.n_data, 1))), axis=1)

        con_outputs = np.dot(con_inputs, self.weights)
        num_classes = np.shape(con_targets)[1]

        if num_classes == 1:
            num_classes = 2
            con_outputs = np.where(con_outputs > 0, 1, 0)
        else:
            # 1 of N encoding
            con_outputs = np.argmax(con_outputs, 1)
            con_targets = np.argmax(con_targets, 1)

        if verbose:
            print("Outputs at the end of the iterations are: ")
            print(np.transpose(con_outputs[0:5, :]))
            print("The targets for these were: ")
            print(np.transpose(con_targets[0:5, :]))
            print()

        conf_mat = np.zeros((num_classes, num_classes))
        for i in range(num_classes):
            for j in range(num_classes):
                conf_mat[i, j] = np.sum(np.where(con_outputs == i, 1, 0) * np.where(con_targets == j, 1, 0))

        print(conf_mat)
        print(np.trace(conf_mat) / np.sum(conf_mat))
        print("----------------------------------------------")

        return con_outputs

if __name__ == '__main__':
    """ 
        Run the OR, AND and the XOR logic functions through the perceptron
        and see if it is able to train the outputs to match the 
        3 functions given x1, x2, and y for 4 different datapoints
    """
    # OR
    a = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    # AND
    b = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]])
    # XOR
    c = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])

    # OR function
    print("Running the OR function through the perceptron")
    p = Perceptron(a[:, 0:2], a[:, 2:])
    p.train_step(a[:, 0:2], a[:, 2:], 0.25, 20, verbose=True)
    p.confusion_matrix(a[:, 0:2], a[:, 2:], verbose=True)

    # AND function
    print("Running the AND function through the perceptron")
    q = Perceptron(b[:, 0:2], b[:, 2:])
    q.train_step(b[:, 0:2], b[:, 2:], 0.25, 20, verbose=True)
    q.confusion_matrix(b[:, 0:2], b[:, 2:], verbose=True)

    # XOR function
    print("Running the XOR function through the perceptron")
    r = Perceptron(c[:, 0:2], c[:, 2:])
    r.train_step(c[:, 0:2], c[:, 2:], 0.25, 20, verbose=True)
    r.confusion_matrix(c[:, 0:2], c[:, 2:], verbose=True)
