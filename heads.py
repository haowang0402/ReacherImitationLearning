import torch
import torch.nn as nn
import torch.nn.functional as F
import ConfigDict
class MLP(nn.Module):
    """Baseline of Multilayer perceptron.
    Attributes:
        input_size (int): size of input
        output_size (int): size of output layer
        hidden_sizes (list): sizes of hidden layers
        hidden_activation (function): activation function of hidden layers
        output_activation (function): activation function of output layer
        hidden_layers (list): list containing linear layers
        use_output_layer (bool): whether or not to use the last layer
        n_category (int): category number (-1 if the action is continuous)
    """

    def __init__(
        self,
        configs: ConfigDict,
        hidden_activation: Callable = F.relu,
        linear_layer: nn.Module = nn.Linear,
        use_output_layer: bool = True,
        n_category: int = -1,
        init_fn: Callable = init_layer_uniform,
    ):
        """Initialize."""
        super(MLP, self).__init__()

        self.hidden_sizes = configs.hidden_sizes
        self.input_size = configs.input_size
        self.output_size = configs.output_size
        self.hidden_activation = hidden_activation
        self.output_activation = configs.output_activation
        self.linear_layer = linear_layer
        self.use_output_layer = use_output_layer
        self.n_category = n_category

        # set hidden layers
        self.hidden_layers: list = []
        in_size = self.input_size
        for i, next_size in enumerate(configs.hidden_sizes):
            fc = self.linear_layer(in_size, next_size)
            in_size = next_size
            self.__setattr__("hidden_fc{}".format(i), fc)
            self.hidden_layers.append(fc)

        # set output layers
        if self.use_output_layer:
            self.output_layer = self.linear_layer(in_size, configs.output_size)
            self.output_layer = init_fn(self.output_layer)
        else:
            self.output_layer = identity
            self.output_activation = identity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        for hidden_layer in self.hidden_layers:
            x = self.hidden_activation(hidden_layer(x))
        x = self.output_activation(self.output_layer(x))

        return x
