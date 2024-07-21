import torch
import torch.nn as nn
import torch.nn.functional as F
from op import correlation

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

    return torch.nn.functional.grid_sample(
        input=f,
        grid=(grid - u * dt).permute(0, 2, 3, 1),
        mode='bilinear',
        padding_mode='reflection',
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
        self.C, self.H, self.W = config.data.num_channels, config.data.image_size, config.data.image_size
        self.fln = len(config.model.feature_nums)  # num of feature layers

        feature_extractors = []
        ch_i = self.C
        for i in range(self.fln):
            ch_o = config.model.feature_nums[i]
            feature_extractors.append(get_conv_feature_layer(ch_i, ch_o))
            ch_i = ch_o

        self.feature_extractors = nn.ModuleList(feature_extractors)

    def forward(self, x):
        result = []
        for layer in self.feature_extractors:
            x = layer(x)
            result.append(x)

            if torch.isnan(x).any():
                print('Nan feature')

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

        flow = flow + self.corr_conv(corr)
        if torch.isnan(flow).any():
            print('Nan matching')

        return flow


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
        flow = flow + self.flow_conv(block)
        if torch.isnan(flow).any():
            print('Nan refinement')

        return flow

class PressureInfer(nn.Module):
    def __init__(self, config, level):
        super(PressureInfer, self).__init__()

        block_depth = config.model.feature_nums[level]*2 + 2 + 1  # feature1 + feature2 + flow(2) + flow_norm(1)
        self.flow_conv = get_conv_field_layer(block_depth, 1)

    def forward(self, feature1, feature2, flow, p_prev=None):

        if p_prev is None:
            p_prev = 0.0

        flow = flow.detach()
        flow_norm = (flow ** 2).sum(dim=1).unsqueeze(1)

        block = torch.cat([feature1.detach(), feature2.detach(), flow, flow_norm], dim=1)
        p_prev = p_prev + self.flow_conv(block)

        if torch.isnan(p_prev).any():
            print('Nan pressure')

        return p_prev


class InferenceUnit(nn.Module):
    def __init__(self, config, level):
        super(InferenceUnit, self).__init__()
        self.level = level
        self.match = Matching(config, level)
        self.refinement = SubpixelRefinement(config, level)
        self.p_inference = PressureInfer(config, level)

    def forward(self, feature1, feature2, flow=None, p_prev=None):
        flow_m = self.match(feature1, feature2, flow)
        flow_s = self.refinement(feature1, feature2, flow_m)

        pressure = self.p_inference(feature1, feature2, flow_s, p_prev)
        return flow_s, pressure


class Upsample(nn.Module):
    def __init__(self, size):
        super(Upsample, self).__init__()

        self.up_flow = get_conv_up_layer(2)
        self.up_pres = get_conv_up_layer(1)
        self.size = size

    def forward(self, f1, f2, flow, pres):
        flow = F.interpolate(input=flow, size=self.size, mode='bilinear', align_corners=False)
        pres = F.interpolate(input=pres, size=self.size, mode='bilinear', align_corners=False)

        flow_block = torch.cat([f1, f2, flow], dim=1)
        pres_block = torch.cat([f1, f2, pres], dim=1)

        return flow + self.up_flow(flow_block), pres + self.up_pres(pres_block)




class FlowNet(nn.Module):
    def __init__(self, config):
        super(FlowNet, self).__init__()

        self.size = (config.data.image_size, config.data.image_size)
        self.feature_extractor = FeatureExtractor(config)

        levels = [l for l in range(len(config.model.feature_nums))][::-1] # level n-1, n-2, ..., 0
        self.inference_units = nn.ModuleList([InferenceUnit(config, level) for level in levels])

        self.upsample = Upsample(self.size)

    def forward(self, f1, f2, coord, t):
        f1_features = self.feature_extractor(f1)
        f2_features = self.feature_extractor(f2)
        
        cascaded_flow = []
        cascaded_pressure = []
        flow = None
        pressure = None
        for unit in self.inference_units:
            feature1 = f1_features[unit.level]
            feature2 = f2_features[unit.level]
            flow, pressure = unit(feature1, feature2, flow)
            cascaded_flow.append(flow)
            cascaded_pressure.append(pressure)
        
        flow, pressure = self.upsample(f1, f2, flow, pressure)
        cascaded_flow.append(flow)
        cascaded_pressure.append(pressure)

        return cascaded_flow, cascaded_pressure


if __name__ == '__main__':
    a = torch.randn(1, 1, 224, 224,device='cuda:0')
    b = torch.randn(1, 1, 224, 224,device='cuda:0')

    c = correlation.FunctionCorrelation(a, b, stride=1)
    print(c.shape)