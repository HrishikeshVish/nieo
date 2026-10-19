import math

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import grad

from utils.config import SHINEConfig


class Decoder(nn.Module):
    def __init__(self, config: SHINEConfig, is_geo_encoder = True, is_time_conditioned = False): 
        
        super().__init__()
        
        if is_geo_encoder:
            mlp_hidden_dim = config.geo_mlp_hidden_dim
            mlp_bias_on = config.geo_mlp_bias_on
            mlp_level = config.geo_mlp_level
        else:
            mlp_hidden_dim = config.sem_mlp_hidden_dim
            mlp_bias_on = config.sem_mlp_bias_on
            mlp_level = config.sem_mlp_level

        input_layer_count = config.feature_dim
        if is_time_conditioned:
            input_layer_count += 1

        # predict sdf (now it anyway only predict sdf without further sigmoid
        # Initializa the structure of shared MLP
        layers = []
        for i in range(mlp_level):
            if i == 0:
                layers.append(nn.Linear(input_layer_count, mlp_hidden_dim, mlp_bias_on))
            else:
                layers.append(nn.Linear(mlp_hidden_dim, mlp_hidden_dim, mlp_bias_on))
        self.layers = nn.ModuleList(layers)
        self.lout = nn.Linear(mlp_hidden_dim, 1, mlp_bias_on)
        self.nclass_out = nn.Linear(mlp_hidden_dim, config.sem_class_count + 1, mlp_bias_on) # sem class + free space class
        # self.bn = nn.BatchNorm1d(self.hidden_dim, affine=False)

        # Activation. ReLU is the SHINE default, but every Hessian-based term in
        # utils/loss.py needs a field that is at least C^2: a ReLU MLP is
        # piecewise linear, so its second derivative w.r.t. its own input is
        # zero almost everywhere and whatever Hessian autograd returns comes
        # only from the octree's trilinear feature interpolation -- a quantity
        # that is itself only C^0 across voxel boundaries. [3] GradHess (Wang et
        # al., NeurIPS 2023) states the requirement outright ("our neural
        # function has to be at least C^2 continuous") and uses Sine (SIREN) or
        # SoftPlus for exactly this reason.
        self.activation_type = config.mlp_activation
        self.siren_omega0 = config.siren_omega0
        if self.activation_type == "sine":
            self._siren_init(input_layer_count, mlp_hidden_dim)

        self.to(config.device)
        # torch.cuda.empty_cache()

    def _siren_init(self, input_dim, hidden_dim):
        """SIREN's initialization (Sitzmann et al., NeurIPS 2020, Sec. 3.2).

        The scheme is not optional decoration: sine layers initialized like ReLU
        layers either saturate or collapse to a near-linear map, and the whole
        point of switching activations here is the second derivative. First
        layer draws from U(-1/n, 1/n) because its pre-activation is multiplied
        by omega0; every later layer draws from U(-sqrt(6/n)/omega0,
        +sqrt(6/n)/omega0), which keeps the pre-activation variance invariant
        through depth.

        Note this decoder's input is the interpolated octree feature vector, not
        a raw coordinate, so omega0 sets frequency in *feature* space. The
        published omega0=30 is calibrated for coordinates in [-1, 1]; treat it
        as a starting point to tune, not a constant.
        """
        with torch.no_grad():
            for k, l in enumerate(self.layers):
                fan_in = input_dim if k == 0 else hidden_dim
                if k == 0:
                    bound = 1.0 / fan_in
                else:
                    bound = math.sqrt(6.0 / fan_in) / self.siren_omega0
                l.weight.uniform_(-bound, bound)
                if l.bias is not None:
                    l.bias.uniform_(-bound, bound)
            # the output layer stays linear, so it keeps the standard init

    def _act(self, x, first_layer=False):
        if self.activation_type == "sine":
            # omega0 multiplies the first layer only; later layers absorb it
            # into their initialization bounds above
            return torch.sin(self.siren_omega0 * x) if first_layer else torch.sin(x)
        if self.activation_type == "softplus":
            # [3] GradHess's other C^inf choice, via Neural-Pull. beta=100 keeps
            # it close to ReLU so it is a drop-in for a config tuned on ReLU.
            return F.softplus(x, beta=100)
        return F.relu(x)

    def forward(self, feature):
        # If we use BCEwithLogits loss, do not need to do sigmoid mannually
        output = self.sdf(feature)
        return output

    # predict the sdf (opposite sign to the actual sdf)
    def sdf(self, sum_features):
        for k, l in enumerate(self.layers):
            if k == 0:
                h = self._act(l(sum_features), first_layer=True)
            else:
                h = self._act(l(h))

        out = self.lout(h).squeeze(1)
        # linear (feature_dim -> hidden_dim)
        # relu
        # linear (hidden_dim -> hidden_dim)
        # relu
        # linear (hidden_dim -> 1)

        return out
    
    def time_conditionded_sdf(self, sum_features, ts):
        time_conditioned_feature = torch.torch.cat((sum_features, ts.view(-1, 1)), dim=1)

        for k, l in enumerate(self.layers):
            if k == 0:
                h = self._act(l(time_conditioned_feature), first_layer=True)
            else:
                h = self._act(l(h))

        out = self.lout(h).squeeze(1)
        # linear (feature_dim + 1 -> hidden_dim)
        # relu
        # linear (hidden_dim -> hidden_dim)
        # relu
        # linear (hidden_dim -> 1)

        return out

    # predict the occupancy probability
    def occupancy(self, sum_features):
        out = torch.sigmoid(self.sdf(sum_features))  # to [0, 1]
        return out

    # predict the probabilty of each semantic label
    def sem_label_prob(self, sum_features):
        for k, l in enumerate(self.layers):
            if k == 0:
                h = self._act(l(sum_features), first_layer=True)
            else:
                h = self._act(l(h))

        out = F.log_softmax(self.nclass_out(h), dim=1)
        return out

    def sem_label(self, sum_features):
        out = torch.argmax(self.sem_label_prob(sum_features), dim=1)
        return out
