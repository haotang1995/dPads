import torch, torch.nn as nn
from .library_functions import LibraryFunction, device, AffineFeatureSelectionFunction
from nsp.tasks.arith.perception import ResDigitNet18

class SqueezeList(LibraryFunction):
    def __init__(self, input_size, output_size, num_units, name="SqueezeList"):
        assert(input_size == output_size and output_size == 3)
        super().__init__({}, "list", "atom", input_size, output_size, num_units, name=name, has_params=False,)
    def execute_on_batch(self, batch, batch_lens, is_sequential=False):
        assert len(batch.size()) == 3
        batch_size, seq_len, feature_dim = batch.size()
        assert seq_len == 1
        return batch[:,0,:] # squeeze

SHARED_CNN_LAYERS = {
    (3072, 1,): ResDigitNet18(symbol_num=3, input_image_size=(3, 32, 96), split_image_flag=True, relu_after_fc1_flag=False,).to(device),
}

class CNNFunction(LibraryFunction):
    def __init__(self, input_size, output_size, num_units, name="CNN"):
        assert hasattr(self, "full_feature_dim")
        assert input_size >= self.full_feature_dim
        if self.full_feature_dim == 0:
            self.is_full = True
            self.full_feature_dim = input_size
        else:
            self.is_full = False
        additional_inputs = input_size - self.full_feature_dim

        assert hasattr(self, "feature_tensor")
        assert len(self.feature_tensor) <= input_size
        self.feature_tensor = self.feature_tensor.to(device)
        self.selected_input_size = self.feature_tensor.size()[-1]+additional_inputs
        super().__init__({}, "atom", "atom", input_size, output_size, num_units, name=name, has_params=True)
    def init_params(self):
        self.raw_input_size = self.input_size
        if self.is_full:
            self.full_feature_dim = self.input_size
            self.feature_tensor = torch.arange(self.input_size).to(device)

        additional_inputs = self.raw_input_size - self.full_feature_dim
        self.selected_input_size = self.feature_tensor.size()[-1] + additional_inputs
        self.cnn_layer = SHARED_CNN_LAYERS[(self.selected_input_size, self.output_size,)]
        self.parameters = {
            f"shared_param{pi}_{self.selected_input_size}_{self.output_size}" : p
            for pi, p in enumerate(self.cnn_layer.parameters())
        }
    def execute_on_batch(self, batch, batch_lens=None):
        assert len(batch.size()) == 2
        batch = batch.reshape(batch.size(0), 3, 32, 96)
        batch = batch[:, :, :, 32*self.img_index:32*(self.img_index+1)]
        if batch.device != device:
            cnn_layer = self.cnn_layer.to(batch.device)
        else:
            cnn_layer = self.cnn_layer
        return cnn_layer(batch)

IMGMATH_FEATURE_SUBSETS = {
    'X': (torch.arange(32*32*3), 0),
    'Y': (torch.arange(32*32*3, 32*32*3*2), 1),
    'Z': (torch.arange(32*32*3*2, 32*32*3*3), 2),
}
IMGMATH_FULL_FEATURE_DIM = 32*32*3*3


class ImgMathXSelection(CNNFunction):
    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = IMGMATH_FULL_FEATURE_DIM
        self.feature_tensor, self.img_index = IMGMATH_FEATURE_SUBSETS['X']
        super().__init__(input_size, output_size, num_units, name="X")

class ImgMathYSelection(CNNFunction):
    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = IMGMATH_FULL_FEATURE_DIM
        self.feature_tensor, self.img_index = IMGMATH_FEATURE_SUBSETS['Y']
        super().__init__(input_size, output_size, num_units, name="Y")

class ImgMathZSelection(CNNFunction):
    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = IMGMATH_FULL_FEATURE_DIM
        self.feature_tensor, self.img_index = IMGMATH_FEATURE_SUBSETS['Z']
        super().__init__(input_size, output_size, num_units, name="Z")

from .library_functions import AddFunction, SubFunction, MultiplyFunction
from .neural_functions import HeuristicNeuralFunction, FeedForwardModule

def imgmath_init_neural_function(input_type, output_type, input_size, output_size, num_units):
    if input_type == "atom" and output_type == "atom":
        return ImgMathAtomToAtomModule(input_size, output_size, num_units)
    else:
        raise ValueError(f"imgmath_init_neural_function: input_type={input_type}, output_type={output_type}, input_size={input_size}, output_size={output_size}, num_units={num_units}")
    return None

class ImgMathAtomToAtomModule(HeuristicNeuralFunction):

    def __init__(self, input_size, output_size, num_units):
        super().__init__("atom", "atom", input_size, output_size, num_units, "ImgMathAtomToAtomModule")

    def init_model(self):
        self.backbone = list(SHARED_CNN_LAYERS.values())[0]
        self.model = FeedForwardModule(3, self.output_size, self.num_units).to(device)

    def execute_on_batch(self, batch, batch_lens=None):
        assert len(batch.size()) == 2
        if batch.device !=  device:
            model = self.model.to(batch.device)
            backbone = self.backbone.to(batch.device)
        else:
            model = self.model
            backbone = self.backbone
        batch = batch.reshape(batch.size(0), 3, 32, 96)
        feat = backbone(batch)
        model_out = model(feat)
        assert len(model_out.size()) == 2
        return model_out

class ImgMathAddFunction(AddFunction):
    def __init__(self, input_size, output_size, num_units, function1=None, function2=None,):
        if function1 is None:
            function1 = imgmath_init_neural_function("atom", "atom", input_size, output_size, num_units)
        if function2 is None:
            function2 = imgmath_init_neural_function("atom", "atom", input_size, output_size, num_units)
        super().__init__(input_size, output_size, num_units, function1=function1, function2=function2)

class ImgMathSubFunction(SubFunction):
    def __init__(self, input_size, output_size, num_units, function1=None, function2=None,):
        if function1 is None:
            function1 = imgmath_init_neural_function("atom", "atom", input_size, output_size, num_units)
        if function2 is None:
            function2 = imgmath_init_neural_function("atom", "atom", input_size, output_size, num_units)
        super().__init__(input_size, output_size, num_units, function1=function1, function2=function2)

class ImgMathMultiplyFunction(MultiplyFunction):
    def __init__(self, input_size, output_size, num_units, function1=None, function2=None,):
        if function1 is None:
            function1 = imgmath_init_neural_function("atom", "atom", input_size, output_size, num_units)
        if function2 is None:
            function2 = imgmath_init_neural_function("atom", "atom", input_size, output_size, num_units)
        super().__init__(input_size, output_size, num_units, function1=function1, function2=function2)
