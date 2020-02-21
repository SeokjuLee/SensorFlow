import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as pytorch_models
import pdb

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        #self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        #self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        #out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        #out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def make_layer(inplanes, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(block(inplanes, planes, stride, downsample))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes, planes))

    return nn.Sequential(*layers)

def downsample_conv(in_planes, out_planes, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2),
        nn.ReLU(inplace=True)
    )


def predict_flow(in_planes, out_planes=2):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1),
        # nn.Sigmoid()
    )


def conv(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )


def upconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ReLU(inplace=True)
    )


def crop_like(input, ref):
    assert(input.size(2) >= ref.size(2) and input.size(3) >= ref.size(3))
    return input[:, :, :ref.size(2), :ref.size(3)]

def encode_sensor(in_feats, out_feats):
    return nn.Sequential(
        nn.Linear(in_feats, out_feats),
        nn.ELU(inplace=True)
    )


class _ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(_ConvBlock, self).__init__()

        self.conv = _Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out

class _Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(_Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

def _upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")


class SFResNet50(nn.Module):

    def __init__(self, alpha=20, dim_motion=6):
        super(SFResNet50, self).__init__()

        self.alpha = alpha
        self.ns = dim_motion    # Quaternion (4) + Translation (3)
        self.dim_sen = 32

        self.encoder = pytorch_models.resnet50(pretrained=True)

        ############ [   0     1     2     3     4]
        enc_planes = [  64,  256,  512, 1024, 2048]
        dec_planes = [  16,   32,   64,  128,  256]

        self.upconv4_0 = _ConvBlock(enc_planes[4], dec_planes[4])
        self.upconv4_1 = _ConvBlock(dec_planes[4] + enc_planes[3], dec_planes[4])
        self.upconv3_0 = _ConvBlock(dec_planes[4], dec_planes[3])
        self.upconv3_1 = _ConvBlock(dec_planes[3] + enc_planes[2], dec_planes[3])

        self.upconv2_0_inv = _ConvBlock(dec_planes[3], dec_planes[2])
        self.upconv2_1_inv = _ConvBlock(dec_planes[2] + enc_planes[1], dec_planes[2])
        self.upconv1_0_inv = _ConvBlock(dec_planes[2], dec_planes[1])
        self.upconv1_1_inv = _ConvBlock(dec_planes[1] + enc_planes[0], dec_planes[1])
        self.upconv0_0_inv = _ConvBlock(dec_planes[1], dec_planes[0])
        self.upconv0_1_inv = _ConvBlock(dec_planes[0], dec_planes[0])

        self.upconv2_0_fwd = _ConvBlock(dec_planes[3], dec_planes[2])
        self.upconv2_1_fwd = _ConvBlock(dec_planes[2] + enc_planes[1], dec_planes[2])
        self.upconv1_0_fwd = _ConvBlock(dec_planes[2], dec_planes[1])
        self.upconv1_1_fwd = _ConvBlock(dec_planes[1] + enc_planes[0], dec_planes[1])
        self.upconv0_0_fwd = _ConvBlock(dec_planes[1], dec_planes[0])
        self.upconv0_1_fwd = _ConvBlock(dec_planes[0], dec_planes[0])

        self.predict_flow_inv = _Conv3x3(dec_planes[0], 2)
        self.predict_flow_fwd = _Conv3x3(dec_planes[0], 2)

        self.linear1 = encode_sensor(self.ns, 16)
        self.linear2 = encode_sensor(16, self.dim_sen)

        self.sconv1 = conv(enc_planes[4]+self.dim_sen, enc_planes[4])
        self.sconv2 = conv(enc_planes[4], enc_planes[4])


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x1, x2):
        '''
            x1: image,  bs x (3-ch) x 448 x 704
            x2: motion, bs x (N-ch)
        '''

        """ Encoder """
        self.features = []
        c1 = self.encoder.conv1(x1)
        c1 = self.encoder.bn1(c1)
        self.features.append(self.encoder.relu(c1))                                         # torch.Size([b, 64, 224, 352])
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))  # torch.Size([b, 256, 112, 176])
        self.features.append(self.encoder.layer2(self.features[-1]))                        # torch.Size([b, 512, 56, 88])
        self.features.append(self.encoder.layer3(self.features[-1]))                        # torch.Size([b, 1024, 28, 44])
        self.features.append(self.encoder.layer4(self.features[-1]))                        # torch.Size([b, 2048, 14, 22])

        bb, _, hh, ww = self.features[4].size()

        """ Sensor Modulator """
        enc_sensor1 = self.linear1(x2)
        enc_sensor2 = self.linear2(enc_sensor1)

        lat_sensor = enc_sensor2.view(bb,self.dim_sen,1,1).expand(bb,self.dim_sen,hh,ww)
        lat_cat = torch.cat((self.features[4], lat_sensor), 1)
        emb_sensor1 = self.sconv1(lat_cat)
        emb_sensor2 = self.sconv2(emb_sensor1)      # torch.Size([b, 2048, 14, 22])

        """ Decoder """
        out_conv4_0 = self.upconv4_0(emb_sensor2)
        concat4_0 = torch.cat((_upsample(out_conv4_0), self.features[3]), 1)
        out_conv4_1 = self.upconv4_1(concat4_0)     # torch.Size([b, 256, 28, 44])

        out_conv3_0 = self.upconv3_0(out_conv4_1)
        concat3_0 = torch.cat((_upsample(out_conv3_0), self.features[2]), 1)
        out_conv3_1 = self.upconv3_1(concat3_0)     # torch.Size([b, 128, 56, 88])

        """ Inverse branch """
        out_conv2_0_inv = self.upconv2_0_inv(out_conv3_1)
        concat2_0_inv = torch.cat((_upsample(out_conv2_0_inv), self.features[1]), 1)
        out_conv2_1_inv = self.upconv2_1_inv(concat2_0_inv)                 # torch.Size([b, 64, 112, 176])

        out_conv1_0_inv = self.upconv1_0_inv(out_conv2_1_inv)
        concat1_0_inv = torch.cat((_upsample(out_conv1_0_inv), self.features[0]), 1)
        out_conv1_1_inv = self.upconv1_1_inv(concat1_0_inv)                 # torch.Size([b, 32, 224, 352])
        
        out_conv0_0_inv = self.upconv0_0_inv(out_conv1_1_inv)
        out_conv0_1_inv = self.upconv0_1_inv(_upsample(out_conv0_0_inv))    # torch.Size([b, 16, 448, 704])
        flow_inv = self.alpha * self.predict_flow_inv(out_conv0_1_inv)      # torch.Size([b, 2, 448, 704])

        """ Forward branch """
        out_conv2_0_fwd = self.upconv2_0_fwd(out_conv3_1)
        concat2_0_fwd = torch.cat((_upsample(out_conv2_0_fwd), self.features[1]), 1)
        out_conv2_1_fwd = self.upconv2_1_fwd(concat2_0_fwd)                 # torch.Size([b, 64, 112, 176])

        out_conv1_0_fwd = self.upconv1_0_fwd(out_conv2_1_fwd)
        concat1_0_fwd = torch.cat((_upsample(out_conv1_0_fwd), self.features[0]), 1)
        out_conv1_1_fwd = self.upconv1_1_fwd(concat1_0_fwd)                 # torch.Size([b, 32, 224, 352])
        
        out_conv0_0_fwd = self.upconv0_0_fwd(out_conv1_1_fwd)
        out_conv0_1_fwd = self.upconv0_1_fwd(_upsample(out_conv0_0_fwd))    # torch.Size([b, 16, 448, 704])
        flow_fwd = self.alpha * self.predict_flow_fwd(out_conv0_1_fwd)      # torch.Size([b, 2, 448, 704])
        
        """ Output concat """
        flow = torch.cat((flow_inv, flow_fwd), 1)   # torch.Size([2, 4, 448, 704])
        # pdb.set_trace()

        if self.training:
            return tuple([flow])
        else:
            return flow
