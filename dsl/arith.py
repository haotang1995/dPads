import torch, torch.nn as nn
from .library_functions import LibraryFunction, device, AffineFeatureSelectionFunction

class SqueezeList(LibraryFunction):
    def __init__(self, input_size, output_size, num_units, name="SqueezeList"):
        assert(input_size == output_size and output_size == 3)
        super().__init__({}, "list", "atom", input_size, output_size, num_units, name=name, has_params=False,)
    def execute_on_batch(self, batch, batch_lens, is_sequential=False):
        assert len(batch.size()) == 3
        batch_size, seq_len, feature_dim = batch.size()
        assert seq_len == 1
        return batch[:,0,:] # squeeze

SHARED_LINEAR_LAYERS = {
    (3, 1, True): nn.Linear(3, 1, bias=True).to(device),
    (3, 3, True): nn.Linear(3, 3, bias=True).to(device),
    (1, 1, True): nn.Linear(1, 1, bias=True).to(device),
}

class FeatureSelectionFunction(AffineFeatureSelectionFunction):
    def init_params(self):
        self.raw_input_size = self.input_size
        if self.is_full:
            self.full_feature_dim = self.input_size
            self.feature_tensor = torch.arange(self.input_size).to(device)

        additional_inputs = self.raw_input_size - self.full_feature_dim
        self.selected_input_size = self.feature_tensor.size()[-1] + additional_inputs
        self.linear_layer = SHARED_LINEAR_LAYERS[(self.selected_input_size, self.output_size, True)]
        self.parameters = {
            f"shared_weights_{self.selected_input_size}_{self.output_size}" : self.linear_layer.weight,
            f"shared_bias_{self.selected_input_size}_{self.output_size}" : self.linear_layer.bias
        }

ARITH_FEATURE_SUBSETS = {
    'X': torch.LongTensor([0]),
    'Y': torch.LongTensor([1]),
    'Z': torch.LongTensor([2]),
}
ARITH_FULL_FEATURE_DIM = 3

class ArithXSelection(FeatureSelectionFunction):
    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = ARITH_FULL_FEATURE_DIM
        self.feature_tensor = ARITH_FEATURE_SUBSETS['X']
        super().__init__(input_size, output_size, num_units, name="X")


class ArithYSelection(FeatureSelectionFunction):
    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = ARITH_FULL_FEATURE_DIM
        self.feature_tensor = ARITH_FEATURE_SUBSETS['Y']
        super().__init__(input_size, output_size, num_units, name="Y")


class ArithZSelection(FeatureSelectionFunction):
    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = ARITH_FULL_FEATURE_DIM
        self.feature_tensor = ARITH_FEATURE_SUBSETS['Z']
        super().__init__(input_size, output_size, num_units, name="Z")

# ===========================================================

class XSelection(AffineFeatureSelectionFunction):
    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = ARITH_FULL_FEATURE_DIM
        self.feature_tensor = ARITH_FEATURE_SUBSETS['X']
        super().__init__(input_size, output_size, num_units, name="X")


class YSelection(AffineFeatureSelectionFunction):
    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = ARITH_FULL_FEATURE_DIM
        self.feature_tensor = ARITH_FEATURE_SUBSETS['Y']
        super().__init__(input_size, output_size, num_units, name="Y")


class ZSelection(AffineFeatureSelectionFunction):
    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = ARITH_FULL_FEATURE_DIM
        self.feature_tensor = ARITH_FEATURE_SUBSETS['Z']
        super().__init__(input_size, output_size, num_units, name="Z")
