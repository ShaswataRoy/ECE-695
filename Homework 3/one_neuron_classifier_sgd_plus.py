#!/usr/bin/env python

##  one_neuron_classifier.py

import random
import numpy as np
import matplotlib.pyplot as plt
import operator
seed = 0
random.seed(seed)
np.random.seed(seed)
from ComputationalGraphPrimer import *


class ComputationalGraphPrimerSGDPlus(ComputationalGraphPrimer):
    def __init__(self, *args, **kwargs):
        if 'mu' in kwargs:
            self.mu = kwargs.pop('mu')
        else:
            self.mu = 0
        super().__init__(*args, **kwargs)
        self.prev_step=[]

    def backprop_and_update_params_one_neuron_model(self, y_error,vals_for_input_vars, deriv_sigmoid):
        input_vars = self.independent_vars
        vals_for_input_vars_dict = dict(zip(input_vars,list(vals_for_input_vars)))
        for i, param in enumerate(self.vals_for_learnable_params):
            step = self.mu*self.prev_step[i] + self.learning_rate * y_error *vals_for_input_vars_dict[input_vars[i]] * deriv_sigmoid
            self.vals_for_learnable_params[param] += step
            self.prev_step[i] = step
        self.step_size = self.mu*self.step_size+self.learning_rate*y_error*deriv_sigmoid
        self.bias += self.step_size

    def reset_step(self):
        for i in range(len(self.learnable_params)):
            if len(self.prev_step) < len(self.vals_for_learnable_params):
                self.prev_step.append(0)
            else:
                self.prev_step[i] = 0

    def run_training_loop_one_neuron_model(self, training_data):
        self.vals_for_learnable_params = {param: random.uniform(0,1) for param in self.learnable_params}
        self.bias = random.uniform(0,1)
        self.step_size = 0

        class DataLoader:
            def __init__(self, training_data, batch_size):
                self.training_data = training_data
                self.batch_size = batch_size
                self.class_0_samples = [(item, 0) for item in self.training_data[0]]
                self.class_1_samples = [(item, 1) for item in self.training_data[1]]
            def __len__(self):
                return len(self.training_data[0]) + len(self.training_data[1])
            def _getitem(self):
                cointoss = random.choice([0,1])
                if cointoss == 0:
                    return random.choice(self.class_0_samples)
                else:
                    return random.choice(self.class_1_samples)
            def getbatch(self):
                batch_data,batch_labels = [],[]
                maxval = 0.0
                for _ in range(self.batch_size):
                    item = self._getitem()
                    if np.max(item[0]) > maxval:
                        maxval = np.max(item[0])
                    batch_data.append(item[0])
                    batch_labels.append(item[1])
                batch_data = [item/maxval for item in batch_data]
                batch = [batch_data, batch_labels]
                return batch

        data_loader = DataLoader(training_data, batch_size=self.batch_size)
        loss_running_record = []
        i = 0
        avg_loss_over_literations = 0.0
        self.reset_step()
        for i in range(self.training_iterations):
            data = data_loader.getbatch()
            data_tuples = data[0]
            class_labels = data[1]
            y_preds, deriv_sigmoids =  self.forward_prop_one_neuron_model(data_tuples)
            loss = sum([(abs(class_labels[i] - y_preds[i]))**2 for i in range(len(class_labels))])
            loss_avg = loss / float(len(class_labels))
            avg_loss_over_literations += loss_avg
            if i%(self.display_loss_how_often) == 0:
                avg_loss_over_literations /= self.display_loss_how_often
                loss_running_record.append(avg_loss_over_literations)
                print("[iter=%d]  loss = %.4f" %  (i+1, avg_loss_over_literations))
                avg_loss_over_literations = 0.0
            y_errors = list(map(operator.sub, class_labels, y_preds))
            y_error_avg = sum(y_errors) / float(len(class_labels))
            deriv_sigmoid_avg = sum(deriv_sigmoids) / float(len(class_labels))
            data_tuple_avg = [sum(x) for x in zip(*data_tuples)]
            data_tuple_avg = list(map(operator.truediv, data_tuple_avg,
                                     [float(len(class_labels))] * len(class_labels) ))
            self.backprop_and_update_params_one_neuron_model(y_error_avg, data_tuple_avg, deriv_sigmoid_avg)

        return loss_running_record

if __name__ == '__main__':
    cgp = ComputationalGraphPrimerSGDPlus(
                   one_neuron_model = True,
                   expressions = ['xw=ab*xa+bc*xb+cd*xc+ac*xd'],
                   output_vars = ['xw'],
                   dataset_size = 5000,
                   learning_rate = 1e-3,
    #               learning_rate = 5 * 1e-2,
                   training_iterations = 40000,
                   batch_size = 8,
                   display_loss_how_often = 100,
                   debug = True
          )

    cgp.parse_expressions()
    training_data = cgp.gen_training_data()

    loss_sgd = cgp.run_training_loop_one_neuron_model( training_data )

    cgp_sgd_plus = ComputationalGraphPrimerSGDPlus(
                   one_neuron_model = True,
                   expressions = ['xw=ab*xa+bc*xb+cd*xc+ac*xd'],
                   output_vars = ['xw'],
                   dataset_size = 5000,
                   learning_rate = 1e-3,
    #               learning_rate = 5 * 1e-2,
                   training_iterations = 40000,
                   batch_size = 8,
                   display_loss_how_often = 100,
                   debug = True,
                   mu = 0.99
          )

    cgp_sgd_plus.parse_expressions()
    training_data = cgp_sgd_plus.gen_training_data()

    loss_sgd_plus = cgp_sgd_plus.run_training_loop_one_neuron_model( training_data )

    plt.figure()
    plt.plot(loss_sgd, label = "SGD Training Loss")
    plt.plot(loss_sgd_plus, label = "SGD+ Training Loss")
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.title('SGD+ vs SGD Loss One Neuron')
    plt.legend()
    plt.show()
