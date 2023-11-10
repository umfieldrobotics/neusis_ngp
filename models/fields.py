import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.embedder import get_embedder
import sys

import tinycudann as tcnn
from numpy import log2, exp2

# https://github.com/SuLvXiangXin/zipnerf-pytorch/blob/4de3d21ebb9e15412d36951b56e2d713fddd812b/internal/math.py#L6
def erf(x):
    return torch.sign(x) * torch.sqrt(1 - torch.exp(-4 / torch.pi * x ** 2))

# This implementation is borrowed from Instant-NSR: https://github.com/zhaofuq/Instant-NSR
class SDFNetworkTcnn(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 encoding="hashgrid_tcnn",
                 degree=3,
                 skip_in=[],
                 multires=0,
                 bias=0.5,
                 scale=1,
                 desired_resolution=1024,
                 log2_hashmap_size=19,
                 level_init=4,# initial level of multi-res has encoding
                 steps_per_level=5000, # steps per level of multi-res has encoding
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False):
        super(SDFNetworkTcnn, self).__init__()
        self.scale = scale
        self.include_input=True
        self.n_layers=n_layers
        self.level_init= level_init
        self.steps_per_level= steps_per_level


        per_level_scale = exp2(log2(desired_resolution / 16) / (16 - 1))
        if encoding=="frequency":
            self.encoder, input_ch = get_embedder(multires, input_dims=d_in)
            
        elif encoding=="hashgrid_tcnn":
            self.encoder = tcnn.Encoding(
                n_input_dims=d_in,
                encoding_config={
                    "otype": "HashGrid",
                    "n_levels": 16,
                    "n_features_per_level": 2,
                    "log2_hashmap_size": log2_hashmap_size,
                    "base_resolution": 16,
                    "per_level_scale": per_level_scale,
                },
            )
            input_ch = self.encoder.n_output_dims
            self.hash_encoding_mask = torch.ones(
                16* 2,
                # self.num_levels * self.features_per_level,
                dtype=torch.float32,
            )
            self.resolutions = (torch.from_numpy(np.arange(16, desired_resolution+16,desired_resolution//32, dtype=np.int32))/desired_resolution).view(1,-1)
        else:
            raise NotImplementedError()

        


        sdf_net = []
        for l in range(n_layers):
            if l == 0:
                in_dim = input_ch + d_in if self.include_input else input_ch
  
            else:
                in_dim = d_hidden
            
            if l == n_layers - 1:
                out_dim = d_out
            else:
                out_dim = d_hidden
            
            sdf_net.append(nn.Linear(in_dim, out_dim))

            if geometric_init:
                if l == n_layers - 1:
                    torch.nn.init.normal_(sdf_net[l].weight, mean=np.sqrt(np.pi) / np.sqrt(in_dim), std=0.0001)
                    torch.nn.init.constant_(sdf_net[l].bias, -bias)     

                elif l==0:
                    if self.include_input:
                        torch.nn.init.constant_(sdf_net[l].bias, 0.0)
                        torch.nn.init.normal_(sdf_net[l].weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                        torch.nn.init.constant_(sdf_net[l].weight[:, 3:], 0.0)
                    else:
                        torch.nn.init.constant_(sdf_net[l].bias, 0.0)
                        torch.nn.init.normal_(sdf_net[l].weight[:, :], 0.0, np.sqrt(2) / np.sqrt(out_dim))

                else:
                    torch.nn.init.constant_(sdf_net[l].bias, 0.0)
                    torch.nn.init.normal_(sdf_net[l].weight[:, :], 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                sdf_net[l] = nn.utils.weight_norm(sdf_net[l])

        self.sdf_net = nn.ModuleList(sdf_net)
        self.activation = nn.Softplus(beta=100)

    # https://github.com/autonomousvision/sdfstudio/blob/master/nerfstudio/fields/sdf_field.py
    def update_mask(self, level):
        self.hash_encoding_mask[:] = 1.0
        self.hash_encoding_mask[level * 2:] = 0

    # https://github.com/SuLvXiangXin/zipnerf-pytorch/blob/4de3d21ebb9e15412d36951b56e2d713fddd812b/internal/models.py#L439
    # https://arxiv.org/pdf/2304.06706.pdf Fig 2, downweighting
    def cal_weights(self, r):
        # r and self.resolutions should be both normalized to [~,1]
        self.weights = erf(1/torch.sqrt(8*(r)**2*(self.resolutions.to(r.device))**2))

    def forward(self, inputs, bound=1, use_weights=True):

        inputs = inputs * self.scale
        inputs = inputs.clamp(-bound, bound)

        inputs = (inputs + bound)/(2*bound)
        h = self.encoder(inputs).to(dtype=torch.float)
        
        # mask feature
        h = h * self.hash_encoding_mask.to(h.device)

        # down-weight features according to range and feature resolution
        if use_weights:
            h=h * self.weights

        if self.include_input:
            h = torch.cat([inputs, h], dim=-1)
        for l in range(self.n_layers):
            h = self.sdf_net[l](h)
            if l != self.n_layers - 1:
                h = self.activation(h)
        sdf_output = h
        return sdf_output

    # https://gist.github.com/ventusff/57f47588eaff5f8b77a382260e7da8a3
    def forward_with_nablas(self, x, bound=1):
        with torch.enable_grad():
            x = x.requires_grad_(True)
            sdf = self.forward(x, bound=bound)
            nablas = torch.autograd.grad(
                sdf[:, :1],
                x,
                torch.ones_like(sdf[:, :1], device=x.device),
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]            
        return sdf, nablas

    def sdf(self, x, use_weights=True):
        return self.forward(x, use_weights=use_weights)[:, :1]


# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class SDFNetwork(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=0,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False):
        super(SDFNetwork, self).__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean= np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean= -np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)

    def forward(self, inputs):
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)
            
            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)

    def sdf(self, x):
        return self.forward(x)[:, :1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)

# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class RenderingNetwork(nn.Module):
    def __init__(self,
                 d_feature,
                 mode,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 encoding="frequency",
                 degree=3,
                 scale=1,
                 weight_norm=True,
                 multires_view=0,
                 squeeze_out=True):
        super().__init__()

        self.mode = mode
        self.squeeze_out = squeeze_out
        self.scale = scale
        self.encoding= encoding
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embedview_fn = None
        if multires_view > 0 and self.encoding=="frequency":
            self.embedview_fn, input_ch = get_embedder(multires_view)
            dims[0] += (input_ch - 3)
        elif self.encoding=="sphere_harmonics_tcnn":
            self.embedview_fn = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": degree,
            },
        )
            input_ch = self.embedview_fn.n_output_dims
            dims[0] += (input_ch - 3)
        else:
            raise NotImplementedError()

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()

    def forward(self, points, normals, view_dirs, feature_vectors,bound=1):
        if self.embedview_fn is not None:

            if self.encoding=="sphere_harmonics_tcnn":

                view_dirs = (view_dirs + bound)/(2*bound)

                view_dirs = view_dirs.clamp(0.0, 1.0)

                view_dirs = self.embedview_fn(view_dirs).float()
            else:
                view_dirs = self.embedview_fn(view_dirs)


        rendering_input = None
        points = points * self.scale

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        if self.squeeze_out:
            x = torch.sigmoid(x)
        return x



# This implementation is borrowed from nerf-pytorch: https://github.com/yenchenlin/nerf-pytorch
class NeRF(nn.Module):
    def __init__(self,
                 D=8,
                 W=256,
                 d_in=3,
                 d_in_view=3,
                 multires=0,
                 multires_view=0,
                 output_ch=4,
                 skips=[4],
                 use_viewdirs=False):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.d_in = d_in
        self.d_in_view = d_in_view
        self.input_ch = 3
        self.input_ch_view = 3
        self.embed_fn = None
        self.embed_fn_view = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            self.input_ch = input_ch

        if multires_view > 0:
            embed_fn_view, input_ch_view = get_embedder(multires_view, input_dims=d_in_view)
            self.embed_fn_view = embed_fn_view
            self.input_ch_view = input_ch_view

        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D - 1)])

        ### Implementation according to the official code release
        ### (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_view + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, input_pts, input_views):
        if self.embed_fn is not None:
            input_pts = self.embed_fn(input_pts)
        if self.embed_fn_view is not None:
            input_views = self.embed_fn_view(input_views)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            return alpha, rgb
        else:
            assert False


class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1]) * torch.exp(self.variance * 10.0)
