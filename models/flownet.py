import torch
import torch.nn as nn
import torch.nn.functional as F
from op import correlation, grid_sample
import functools

temp_grid = {}
def project(f, u, dt):
    if str(u.shape) not in temp_grid:
        B, C, H, W = u.shape
        grid_h = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(B, -1, H, -1)
        grid_v = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(B, -1, -1, W)
        temp_grid[str(u.shape)] = torch.cat([grid_h, grid_v], 1)

    grid = temp_grid[str(u.shape)].to(u.device)
    u = torch.cat([
        u[:, 1:2, :, :] / ((f.size(2) - 1.0) / 2.0),
        u[:, 0:1, :, :] / ((f.size(3) - 1.0) / 2.0)
    ], 1)

    return grid_sample.grid_sample_2d(
        input=f,
        grid=(grid - u * dt).permute(0, 2, 3, 1),
        padding_mode='border',
        align_corners=True)

def get_conv_feature_layer(in_channels, out_channels):
    layer = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
    return layer

def get_conv_decode_layer(in_channels, out_channels):
    layer = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
    return layer

def get_conv_field_layer(in_channels, out_channels):
    layer = torch.nn.Sequential(torch.nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=3, stride=1, padding=1),
        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
        torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
        torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
        torch.nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=3, stride=1, padding=1))
    return layer

def get_conv_up_layer(out_channels):
    layer = torch.nn.Sequential(torch.nn.Conv2d(in_channels=2+out_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
        torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
        torch.nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=3, stride=1, padding=1))
    return layer


class FeatureExtractor(nn.Module):
    def __init__(self, config):
        super(FeatureExtractor, self).__init__()
        self.fln = len(config.model.feature_nums)  # num of feature layers

        self.spatial_emb = functools.partial(
            layers.get_spatial_embedding,
            omega=config.model.spatial_embed_omega,
            s=config.model.spatial_embed_s_flow)
        self.semb_down = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        feature_extractors = []
        ch_i = config.data.num_channels
        for i in range(self.fln):
            ch_o = config.model.feature_nums[i]
            feature_extractors.append(get_conv_feature_layer(ch_i, ch_o))
            ch_i = ch_o

        self.feature_extractors = nn.ModuleList(feature_extractors)

    def forward(self, f, x, y, t):
        result = []
        semb = self.spatial_emb(x, y)
        for idx, layer in enumerate(self.feature_extractors):
            channel = f.shape[1]
            temb = layers.get_timestep_embedding(t, channel)[:,:,None,None]
            f = layer(f + semb + temb)
            result.append(f)
            semb = self.semb_down(semb)

        return result


class Matching(nn.Module):
    def __init__(self, config, level):
        super(Matching, self).__init__()
        self.dt = config.data.dt * 0.5**level

        self.flow_upsample = torch.nn.ConvTranspose2d(
                        in_channels=2,
                        out_channels=2,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False,
                        groups=2)

        self.corr_conv = get_conv_field_layer(49, 2)

    def forward(self, feature1, feature2, flow=None):

        # backward warping by previous flow field
        if flow is not None:
            flow = self.flow_upsample(flow)
            feature2 = project(feature2, flow, -self.dt)
        else:
            flow = 0.0

        corr = correlation.FunctionCorrelation(feature1, feature2, stride=1)
        corr = torch.nn.functional.leaky_relu(corr)

        return flow + self.corr_conv(corr)

class SubpixelRefinement(nn.Module):
    def __init__(self, config, level):
        super(SubpixelRefinement, self).__init__()

        self.dt = config.data.dt * 0.5 ** (level+1)

        block_depth = config.model.feature_nums[level]*2 + 2  # feature1 + feature2 + flow(2)
        self.flow_conv = get_conv_field_layer(block_depth, 2)

    def forward(self, feature1, feature2, flow):

        # backward warping by vm
        feature2 = project(feature2, flow, -self.dt)

        block = torch.cat([feature1, feature2, flow], dim=1)
        return flow + self.flow_conv(block)

class InferenceUnit(nn.Module):
    def __init__(self, config, level):
        super(InferenceUnit, self).__init__()
        self.level = level
        self.match = Matching(config, level)
        self.refinement = SubpixelRefinement(config, level)

    def forward(self, feature1, feature2, flow=None, p_prev=None):
        flow_m = self.match(feature1, feature2, flow)
        flow_s = self.refinement(feature1, feature2, flow_m)
        return flow_s


class Upsample(nn.Module):
    def __init__(self):
        super(Upsample, self).__init__()

        self.up = get_conv_up_layer(2)

    def forward(self, f1, f2, x, size):
        x = F.interpolate(input=x, size=size, mode='bilinear', align_corners=False)
        block = torch.cat([f1, f2, x], dim=1)

        return x + self.up(block)


class FlowNet(nn.Module):
    def __init__(self, config):
        super(FlowNet, self).__init__()

        self.size = (config.data.image_size, config.data.image_size)
        self.feature_extractor = FeatureExtractor(config)

        levels = [l for l in range(len(config.model.feature_nums))][::-1] # level n-1, n-2, ..., 0
        self.inference_units = nn.ModuleList([InferenceUnit(config, level) for level in levels])

        self.upsample = Upsample()

    def forward(self, f1, f2, x, y, t, size=None):
        f1_features = self.feature_extractor(f1, x, y, t)
        f2_features = self.feature_extractor(f2, x, y, t)
        
        cascaded_flow = []
        flow = None
        for unit in self.inference_units:
            feature1 = f1_features[unit.level]
            feature2 = f2_features[unit.level]
            flow= unit(feature1, feature2, flow)
            cascaded_flow.append(flow)
        
        flow = self.upsample(f1, f2, flow, self.size if size is None else size)
        cascaded_flow.append(flow)

        return cascaded_flow

    def multiscale_data_mse(self, veloc_pred: list[torch.Tensor], target, error_fn=torch.nn.MSELoss()):
        h, w = veloc_pred[-1].shape[-2], veloc_pred[-1].shape[-1]

        weights = [12.7, 5.5, 4.35, 3.9, 3.4, 1.1][:len(veloc_pred)]

        v_loss = 0
        for i, weight in enumerate(weights):
            scale_factor = 1.0 / (2 ** i)

            flow = veloc_pred[-1 - i]
            losses_flow = error_fn(flow * scale_factor, target[:, :2] * scale_factor)

            v_l = weight * losses_flow

            v_loss += v_l

            h = h // 2
            w = w // 2

            target = F.interpolate(target, (h, w), mode='bilinear', align_corners=False)

        return v_loss

from . import layers
def get_double_res(in_channels, out_channels, num_groups=16):
    layer = nn.Sequential(
        layers.ResidualBlock(in_channels, in_channels*2),
        layers.ResidualBlock(in_channels*2, out_channels)
    )
    return layer
def get_down_layer(in_channels, out_channels):
    layer = nn.Sequential(
        nn.MaxPool2d(2),
        get_double_res(in_channels, out_channels)
    )
    return layer
def get_up_layer(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        )


class PressureNet(nn.Module):
    def __init__(self, config):
        super(PressureNet, self).__init__()

        self.channels = channels = config.model.feature_nums
        self.flow_feature_nums = flow_feature_nums = 32
        self.flow_feature = get_double_res(3, flow_feature_nums)
        self.spatial_emb = functools.partial(
            layers.get_spatial_embedding,
            omega=config.model.spatial_embed_omega,
            s=config.model.spatial_embed_s_pres)
        self.semb_down = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.first = get_double_res(flow_feature_nums, channels[0])

        ch_i = channels[0]
        self.down = []
        for ch_o in channels[1:]:
            self.down.append(get_down_layer(ch_i, ch_o))
            ch_i = ch_o
        self.down = nn.ModuleList(self.down)

        ch_i = channels[-1]
        self.up = []
        self.up_conv = []
        for ch_o in channels[-2::-1]:
            self.up.append(get_up_layer(ch_i, ch_o))
            self.up_conv.append(get_double_res(ch_o*2 + flow_feature_nums, ch_o, 4))
            ch_i = ch_o
        self.up = nn.ModuleList(self.up)
        self.up_conv = nn.ModuleList(self.up_conv)

        self.end = nn.Sequential(
            get_double_res(channels[0], channels[0] // 2),
            nn.Conv2d(channels[0] // 2, channels[0] // 2, kernel_size=1),
            get_double_res(channels[0] // 2, 1),
            nn.Conv2d(1, 1, kernel_size=1)
        )

    def get_norm_feature(self, flow):
        flow_norm = -(flow ** 2).sum(dim=1).unsqueeze(1)
        block = torch.cat([flow, flow_norm], dim=1)
        return self.flow_feature(block)

    def get_semb_list(self, x, y):
        semb = self.spatial_emb(x, y)
        semb_list = [semb]

        for i in range(len(self.channels)-2):
            semb = self.semb_down(semb)
            semb_list.append(semb)

        return semb_list

    def forward(self, cascaded_flow, x, y, t):

        temb = layers.get_timestep_embedding(t, self.flow_feature_nums)[:,:,None,None]
        semb = self.get_semb_list(x, y)

        x = self.get_norm_feature(cascaded_flow[-1].detach().clone()) + temb + semb[0]
        x = self.first(x)
        features = [x]


        for down in self.down:
            x = down(x)
            features.append(x)
        features.pop(-1)

        for idx in range(len(features)):
            feature = features[-1-idx]
            flow_feature = self.get_norm_feature(cascaded_flow[idx+2].detach().clone()) + temb + semb[-1-idx]

            up = self.up[idx]
            up_conv = self.up_conv[idx]

            x = up(x)
            block = torch.cat([feature, x, flow_feature], dim=1)
            x = up_conv(block)

        x = self.end(x)
        return x

    def data_mse(self, pressure, target, error_fn=torch.nn.MSELoss()):
        return error_fn(pressure, target[:,2:3])


if __name__ == '__main__':
    a = torch.randn(1, 1, 224, 224,device='cuda:0')
    b = torch.randn(1, 1, 224, 224,device='cuda:0')

    c = correlation.FunctionCorrelation(a, b, stride=1)
    print(c.shape)