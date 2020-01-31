import torch
import torch.nn as nn
import torch.nn.functional as F
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


def predict_flow(in_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, 2, kernel_size=3, padding=1),
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


class SFResNet(nn.Module):

    def __init__(self, alpha=20, dim_motion=7):
        super(SFResNet, self).__init__()

        self.alpha = alpha
        self.ns = dim_motion     # Quaternion (4) + Translation (3)

        conv_planes = [32, 64, 128, 256, 512, 512, 512]
        self.conv1 = downsample_conv(3,              conv_planes[0], kernel_size=7)
        self.conv2 = make_layer(conv_planes[0], BasicBlock, conv_planes[1], blocks=2, stride=2)
        self.conv3 = make_layer(conv_planes[1], BasicBlock, conv_planes[2], blocks=2, stride=2)
        self.conv4 = make_layer(conv_planes[2], BasicBlock, conv_planes[3], blocks=3, stride=2)
        self.conv5 = make_layer(conv_planes[3], BasicBlock, conv_planes[4], blocks=3, stride=2)
        self.conv6 = make_layer(conv_planes[4], BasicBlock, conv_planes[5], blocks=3, stride=2)
        self.conv7 = make_layer(conv_planes[5], BasicBlock, conv_planes[6], blocks=3, stride=2)

        upconv_planes = [512, 512, 256, 128, 64, 32, 16]
        self.upconv7 = upconv(conv_planes[6],   upconv_planes[0])
        self.upconv6 = upconv(upconv_planes[0], upconv_planes[1])
        self.upconv5 = upconv(upconv_planes[1], upconv_planes[2])
        self.upconv4 = upconv(upconv_planes[2], upconv_planes[3])
        self.upconv3 = upconv(upconv_planes[3], upconv_planes[4])
        self.upconv2 = upconv(upconv_planes[4], upconv_planes[5])
        self.upconv1 = upconv(upconv_planes[5], upconv_planes[6])

        self.jconv7 = make_layer(upconv_planes[0] + conv_planes[5], BasicBlock, upconv_planes[0], blocks=2, stride=1)
        self.jconv6 = make_layer(upconv_planes[1] + conv_planes[4], BasicBlock, upconv_planes[1], blocks=2, stride=1)
        self.jconv5 = make_layer(upconv_planes[2] + conv_planes[3], BasicBlock, upconv_planes[2], blocks=2, stride=1)
        self.jconv4 = make_layer(upconv_planes[3] + conv_planes[2], BasicBlock, upconv_planes[3], blocks=2, stride=1)
        self.jconv3 = make_layer(2 + upconv_planes[4] + conv_planes[1], BasicBlock, upconv_planes[4], blocks=1, stride=1)
        self.jconv2 = make_layer(2 + upconv_planes[5] + conv_planes[0], BasicBlock, upconv_planes[5], blocks=1, stride=1)
        self.jconv1 = make_layer(2 + upconv_planes[6], BasicBlock, upconv_planes[6], blocks=1, stride=1)

        self.predict_flow6 = predict_flow(upconv_planes[1])
        self.predict_flow5 = predict_flow(upconv_planes[2])
        self.predict_flow4 = predict_flow(upconv_planes[3])
        self.predict_flow3 = predict_flow(upconv_planes[4])
        self.predict_flow2 = predict_flow(upconv_planes[5])
        self.predict_flow1 = predict_flow(upconv_planes[6])

        self.linear1 = encode_sensor(self.ns, 16)
        self.linear2 = encode_sensor(16, 32)

        self.sconv1 = conv(conv_planes[6]+32, conv_planes[6])
        self.sconv2 = conv(conv_planes[6], conv_planes[6])


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
        out_conv1 = self.conv1(x1)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)

        bb, _, hh, ww = out_conv7.size()

        """ Sensor Modulator """
        enc_sensor1 = self.linear1(x2)
        enc_sensor2 = self.linear2(enc_sensor1)

        # lat_sensor = enc_sensor2.resize(bb,32,1,1).expand(bb,32,hh,ww)
        lat_sensor = enc_sensor2.view(bb,32,1,1).expand(bb,32,hh,ww)
        lat_cat = torch.cat((out_conv7, lat_sensor), 1)
        emb_sensor1 = self.sconv1(lat_cat)
        emb_sensor2 = self.sconv2(emb_sensor1)
        # pdb.set_trace()

        """ Decoder """
        # out_upconv7 = crop_like(self.upconv7(out_conv7), out_conv6)
        out_upconv7 = crop_like(self.upconv7(emb_sensor2), out_conv6)
        concat7 = torch.cat((out_upconv7, out_conv6), 1)
        out_jconv7 = self.jconv7(concat7)

        out_upconv6 = crop_like(self.upconv6(out_jconv7), out_conv5)
        concat6 = torch.cat((out_upconv6, out_conv5), 1)
        out_jconv6 = self.jconv6(concat6)
        flow6 = self.alpha * self.predict_flow6(out_jconv6)

        out_upconv5 = crop_like(self.upconv5(out_jconv6), out_conv4)
        concat5 = torch.cat((out_upconv5, out_conv4), 1)
        out_jconv5 = self.jconv5(concat5)
        flow5 = self.alpha * self.predict_flow5(out_jconv5)

        out_upconv4 = crop_like(self.upconv4(out_jconv5), out_conv3)
        concat4 = torch.cat((out_upconv4, out_conv3), 1)
        out_jconv4 = self.jconv4(concat4)
        flow4 = self.alpha * self.predict_flow4(out_jconv4)

        out_upconv3 = crop_like(self.upconv3(out_jconv4), out_conv2)
        flow4_up = crop_like(F.interpolate(flow4, scale_factor=2, mode='bilinear', align_corners=True), out_conv2)
        concat3 = torch.cat((out_upconv3, out_conv2, flow4_up), 1)
        out_jconv3 = self.jconv3(concat3)
        flow3 = self.alpha * self.predict_flow3(out_jconv3)

        out_upconv2 = crop_like(self.upconv2(out_jconv3), out_conv1)
        flow3_up = crop_like(F.interpolate(flow3, scale_factor=2, mode='bilinear', align_corners=True), out_conv1)
        concat2 = torch.cat((out_upconv2, out_conv1, flow3_up), 1)
        out_jconv2 = self.jconv2(concat2)
        flow2 = self.alpha * self.predict_flow2(out_jconv2)

        out_upconv1 = crop_like(self.upconv1(out_jconv2), x1)
        flow2_up = crop_like(F.interpolate(flow2, scale_factor=2, mode='bilinear', align_corners=True), x1)
        concat1 = torch.cat((out_upconv1, flow2_up), 1)
        out_jconv1 = self.jconv1(concat1)
        flow1 = self.alpha * self.predict_flow1(out_jconv1)

        if self.training:
            return flow1, flow2, flow3, flow4, flow5, flow6
        else:
            return flow1
