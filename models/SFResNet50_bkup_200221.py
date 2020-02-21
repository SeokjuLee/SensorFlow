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


class SFResNet50(nn.Module):

    def __init__(self, alpha=20, dim_motion=6):
        super(SFResNet50, self).__init__()

        self.alpha = alpha
        self.ns = dim_motion    # Quaternion (4) + Translation (3)
        self.dim_sen = 32

        self.encoder = pytorch_models.resnet50(pretrained=True)

        conv_planes   = [  64,  256,  512, 1024, 2048]
        upconv_planes = [1024,  512,  256,   64,   32]

        self.upconv5 = upconv(conv_planes[4],   upconv_planes[0])
        self.upconv4 = upconv(upconv_planes[0], upconv_planes[1])

        self.upconv3_inv = upconv(upconv_planes[1], upconv_planes[2])
        self.upconv2_inv = upconv(upconv_planes[2], upconv_planes[3])
        self.upconv1_inv = upconv(upconv_planes[3], upconv_planes[4])

        self.upconv3_fwd = upconv(upconv_planes[1], upconv_planes[2])
        self.upconv2_fwd = upconv(upconv_planes[2], upconv_planes[3])
        self.upconv1_fwd = upconv(upconv_planes[3], upconv_planes[4])


        self.iconv5 = make_layer(upconv_planes[0] + conv_planes[3], BasicBlock, upconv_planes[0], blocks=2, stride=1)
        self.iconv4 = make_layer(upconv_planes[1] + conv_planes[2], BasicBlock, upconv_planes[1], blocks=2, stride=1)

        self.iconv3_inv = make_layer(upconv_planes[2] + conv_planes[1], BasicBlock, upconv_planes[2], blocks=2, stride=1)
        self.iconv2_inv = make_layer(upconv_planes[3] + conv_planes[0], BasicBlock, upconv_planes[3], blocks=2, stride=1)
        self.iconv1_inv = make_layer(upconv_planes[4], BasicBlock, upconv_planes[4], blocks=1, stride=1)

        self.iconv3_fwd = make_layer(upconv_planes[2] + conv_planes[1], BasicBlock, upconv_planes[2], blocks=2, stride=1)
        self.iconv2_fwd = make_layer(upconv_planes[3] + conv_planes[0], BasicBlock, upconv_planes[3], blocks=2, stride=1)
        self.iconv1_fwd = make_layer(upconv_planes[4], BasicBlock, upconv_planes[4], blocks=1, stride=1)

        self.predict_flow_inv = predict_flow(upconv_planes[4], 2)
        self.predict_flow_fwd = predict_flow(upconv_planes[4], 2)


        self.linear1 = encode_sensor(self.ns, 16)
        self.linear2 = encode_sensor(16, self.dim_sen)

        self.sconv1 = conv(conv_planes[4]+self.dim_sen, conv_planes[4])
        self.sconv2 = conv(conv_planes[4], conv_planes[4])


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
        emb_sensor2 = self.sconv2(emb_sensor1)  # torch.Size([b, 2048, 14, 22])

        """ Decoder """
        out_upconv5 = crop_like(self.upconv5(emb_sensor2), self.features[3])
        concat5 = torch.cat((out_upconv5, self.features[3]), 1)
        out_iconv5 = self.iconv5(concat5)       # torch.Size([b, 1024, 28, 44])

        out_upconv4 = crop_like(self.upconv4(out_iconv5), self.features[2])
        concat4 = torch.cat((out_upconv4, self.features[2]), 1)
        out_iconv4 = self.iconv4(concat4)       # torch.Size([b, 512, 56, 88])

        """ Inverse branch """
        out_upconv3_inv = crop_like(self.upconv3_inv(out_iconv4), self.features[1])
        concat3_inv = torch.cat((out_upconv3_inv, self.features[1]), 1)
        out_iconv3_inv = self.iconv3_inv(concat3_inv)                   # torch.Size([b, 256, 112, 176])

        out_upconv2_inv = crop_like(self.upconv2_inv(out_iconv3_inv), self.features[0])
        concat2_inv = torch.cat((out_upconv2_inv, self.features[0]), 1)
        out_iconv2_inv = self.iconv2_inv(concat2_inv)                   # torch.Size([b, 64, 224, 352])

        out_upconv1_inv = crop_like(self.upconv1_inv(out_iconv2_inv), x1)
        out_iconv1_inv = self.iconv1_inv(out_upconv1_inv)               # torch.Size([b, 32, 448, 704])
        flow_inv = self.alpha * self.predict_flow_inv(out_iconv1_inv)   # torch.Size([b, 2, 448, 704])

        """ Forward branch """
        out_upconv3_fwd = crop_like(self.upconv3_fwd(out_iconv4), self.features[1])
        concat3_fwd = torch.cat((out_upconv3_fwd, self.features[1]), 1)
        out_iconv3_fwd = self.iconv3_fwd(concat3_fwd)                   # torch.Size([b, 256, 112, 176])

        out_upconv2_fwd = crop_like(self.upconv2_fwd(out_iconv3_fwd), self.features[0])
        concat2_fwd = torch.cat((out_upconv2_fwd, self.features[0]), 1)
        out_iconv2_fwd = self.iconv2_fwd(concat2_fwd)                   # torch.Size([b, 64, 224, 352])

        out_upconv1_fwd = crop_like(self.upconv1_fwd(out_iconv2_fwd), x1)
        out_iconv1_fwd = self.iconv1_fwd(out_upconv1_fwd)               # torch.Size([b, 32, 448, 704])
        flow_fwd = self.alpha * self.predict_flow_fwd(out_iconv1_fwd)   # torch.Size([b, 2, 448, 704])

        """ Output concat """
        flow = torch.cat((flow_inv, flow_fwd), 1)    # torch.Size([2, 4, 448, 704])
        # pdb.set_trace()

        if self.training:
            return tuple([flow])
        else:
            return flow
