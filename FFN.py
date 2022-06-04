import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, hidden_layers, drop_p=0.5):
        super().__init__()
        # TODO: Remove the *9, it's for OHE
        input_size = 81 * 9
        # 81 * 9 Because there are 9 possible outcomes for each position in the board.
        output_size = 81 * 9
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.drop_p = drop_p
        self.sequential = nn.Sequential()
        self.sequential.add_module("input", nn.Linear(input_size, hidden_layers[0]))
        # input_output = zip([layer for layer in hidden_layers[:-1]],
        #                    [layer for layer in hidden_layers[1:]])
        # i = 0
        # for input, output in input_output:
        #     self.sequential.add_module("hidden_{}".format(i), nn.Linear(input, output))
            # self.sequential.add_module("dropout_{}".format(i), nn.Dropout(drop_p))
            # i += 1

        # self.sequential.add_module("output", nn.Linear(hidden_layers[-1], output_size))

    def forward(self, x):
        x = x.view(x.size(0), -1).float()
        before_softmax = self.sequential(x)
        return before_softmax.reshape(x.size(0), 81, 9)
        # output = torch.softmax(before_softmax, dim=2)
        # output = torch.argmax(output, dim=2)+1
        # return output