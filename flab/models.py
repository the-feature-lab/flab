import torch.nn as nn
import torch
import copy

class CenteredWrapper(nn.Module):
    """
    Wrap any nn.Module so that forward(x) returns model(x) - baseline(x),
    where baseline is a frozen snapshot taken at wrap time.
    """
    def __init__(self, model: nn.Module, baseline_dtype=None):
        super().__init__()
        self.model = model
        self.baseline = copy.deepcopy(model)  # snapshot at wrap time
        for p in self.baseline.parameters():
            p.requires_grad = False
        self.baseline.eval()
        self._baseline_dtype = baseline_dtype
        if baseline_dtype is not None:
            self.baseline.to(dtype=baseline_dtype)

    @torch.no_grad()
    def recenter(self):
        """Reset baseline to current model weights (still frozen)."""
        self.baseline.load_state_dict(self.model.state_dict())
        if self._baseline_dtype is not None:
            self.baseline.to(dtype=self._baseline_dtype)
        self.baseline.eval()

    def forward(self, x):
        y = self.model(x)  # grads here
        with torch.inference_mode():
            y0 = self.baseline(x)  # no grads / low mem
        return y - y0


def centeredmodel(model: nn.Module, baseline_dtype=None) -> nn.Module:
    """
    Usage:
        centerednet = centeredmodel(baseline_net)     # wrap for centering
    """
    return CenteredWrapper(model, baseline_dtype=baseline_dtype)


class MLP(nn.Module):
    def __init__(self, d_in=1, width=4096, depth=2, d_out=1, bias=True, **kwargs):
        super().__init__()
        self.d_in, self.width, self.depth, self.d_out = d_in, width, depth, d_out

        self.input_layer = nn.Linear(d_in, width, bias)
        self.hidden_layers = nn.ModuleList([nn.Linear(width, width, bias) for _ in range(depth - 1)])
        self.output_layer = nn.Linear(width, d_out, bias)
        if kwargs.get('zero_output_layer', False):
            with torch.no_grad():
                self.output_layer.weight.zero_()
                if self.output_layer.bias is not None:
                    self.output_layer.bias.zero_()
        self.nonlin = kwargs.get('nonlinearity', nn.ReLU)()
        
    def forward(self, x):
        h = self.nonlin(self.input_layer(x))
        for layer in self.hidden_layers:
            h = self.nonlin(layer(h))
        out = self.output_layer(h)
        return out

    def get_activations(self, x):
        h_acts = []
        h = self.nonlin(self.input_layer(x))
        h_acts.append(h)
        for layer in self.hidden_layers:
            h = self.nonlin(layer(h))
            h_acts.append(h)
        h_out = self.output_layer(h)
        return h_acts, h_out


class CNN(nn.Module):
    def __init__(self, in_channels=1, img_size=8, width=16, depth=2, d_out=1, bias=True):
        super().__init__()
        if depth < 2:
            raise ValueError("CNN depth must be >= 2")
        self.in_channels = in_channels
        self.img_size = img_size
        self.width = width
        self.depth = depth
        self.d_out = d_out

        self.input_layer = nn.Conv2d(in_channels, width, kernel_size=3, padding=1, bias=bias)
        self.hidden_layers = nn.ModuleList(
            [nn.Conv2d(width, width, kernel_size=3, padding=1, bias=bias) for _ in range(depth - 1)]
        )
        self.output_layer = nn.Linear(width * img_size * img_size, d_out, bias=bias)
        self.nonlin = nn.ReLU()

    def forward(self, x):
        h = self.nonlin(self.input_layer(x))
        for layer in self.hidden_layers:
            h = self.nonlin(layer(h))
        h = h.flatten(start_dim=1)
        return self.output_layer(h)


class ExpanderMLP(nn.Module):
    def __init__(self, d_in=1, width=4096, d_out=1, bias=False, nonlinearity=None, forcezeros=False):
        super().__init__()
        self.d_in, self.width, self.d_out = d_in, width, d_out

        self.input_layer = nn.Linear(d_in, d_in, bias)
        self.hidden_layer = nn.Linear(d_in, width, bias)
        self.hidden_layer.weight.requires_grad_(False)
        if self.hidden_layer.bias is not None:
            self.hidden_layer.bias.requires_grad_(False)
        self.output_layer = nn.Linear(width, d_out, bias)
        if forcezeros:
            with torch.no_grad():
                self.output_layer.weight.zero_()
                if self.output_layer.bias is not None:
                    self.output_layer.bias.zero_()
        self.nonlin = nonlinearity if nonlinearity is not None else nn.ReLU()
        
    def forward(self, x):
        h = self.input_layer(x)
        h = self.nonlin(self.hidden_layer(h))
        out = self.output_layer(h)
        return out

    def get_activations(self, x):
        h_acts = []
        h = self.input_layer(x)
        h_acts.append(h)
        h = self.hidden_layer(h)
        h_acts.append(h)
        h_out = self.output_layer(h)
        return h_acts, h_out
