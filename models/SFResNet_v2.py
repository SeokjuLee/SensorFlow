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


class SFResNet_v2(nn.Module):

    def __init__(self, alpha=20, dim_motion=7, ch_pred=2):
        super(SFResNet_v2, self).__init__()

        self.alpha = alpha
        self.ns = dim_motion    # Quaternion (4) + Translation (3)
        self.ch = ch_pred       # predicted channels (x-, y-axis + two-way)?

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
        
        self.upconv3_inv = upconv(upconv_planes[3], upconv_planes[4])
        self.upconv2_inv = upconv(upconv_planes[4], upconv_planes[5])
        self.upconv1_inv = upconv(upconv_planes[5], upconv_planes[6])
        
        self.upconv3_fwd = upconv(upconv_planes[3], upconv_planes[4])
        self.upconv2_fwd = upconv(upconv_planes[4], upconv_planes[5])
        self.upconv1_fwd = upconv(upconv_planes[5], upconv_planes[6])


        self.jconv7 = make_layer(upconv_planes[0] + conv_planes[5], BasicBlock, upconv_planes[0], blocks=2, stride=1)
        self.jconv6 = make_layer(upconv_planes[1] + conv_planes[4], BasicBlock, upconv_planes[1], blocks=2, stride=1)
        self.jconv5 = make_layer(upconv_planes[2] + conv_planes[3], BasicBlock, upconv_planes[2], blocks=2, stride=1)
        self.jconv4 = make_layer(upconv_planes[3] + conv_planes[2], BasicBlock, upconv_planes[3], blocks=2, stride=1)
        
        self.jconv3_inv = make_layer(2 + upconv_planes[4] + conv_planes[1], BasicBlock, upconv_planes[4], blocks=1, stride=1)
        self.jconv2_inv = make_layer(2 + upconv_planes[5] + conv_planes[0], BasicBlock, upconv_planes[5], blocks=1, stride=1)
        self.jconv1_inv = make_layer(2 + upconv_planes[6], BasicBlock, upconv_planes[6], blocks=1, stride=1)
        
        self.jconv3_fwd = make_layer(2 + upconv_planes[4] + conv_planes[1], BasicBlock, upconv_planes[4], blocks=1, stride=1)
        self.jconv2_fwd = make_layer(2 + upconv_planes[5] + conv_planes[0], BasicBlock, upconv_planes[5], blocks=1, stride=1)
        self.jconv1_fwd = make_layer(2 + upconv_planes[6], BasicBlock, upconv_planes[6], blocks=1, stride=1)


        self.predict_flow4_inv = predict_flow(upconv_planes[3], 2)
        self.predict_flow3_inv = predict_flow(upconv_planes[4], 2)
        self.predict_flow2_inv = predict_flow(upconv_planes[5], 2)
        self.predict_flow1_inv = predict_flow(upconv_planes[6], 2)

        self.predict_flow4_fwd = predict_flow(upconv_planes[3], 2)
        self.predict_flow3_fwd = predict_flow(upconv_planes[4], 2)
        self.predict_flow2_fwd = predict_flow(upconv_planes[5], 2)
        self.predict_flow1_fwd = predict_flow(upconv_planes[6], 2)


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

        out_upconv5 = crop_like(self.upconv5(out_jconv6), out_conv4)
        concat5 = torch.cat((out_upconv5, out_conv4), 1)
        out_jconv5 = self.jconv5(concat5)

        out_upconv4 = crop_like(self.upconv4(out_jconv5), out_conv3)
        concat4 = torch.cat((out_upconv4, out_conv3), 1)
        out_jconv4 = self.jconv4(concat4)
        flow4_inv = self.alpha * self.predict_flow4_inv(out_jconv4)
        flow4_fwd = self.alpha * self.predict_flow4_fwd(out_jconv4)

        """ Inverse branch """
        out_upconv3_inv = crop_like(self.upconv3_inv(out_jconv4), out_conv2)
        flow4_up_inv = crop_like(F.interpolate(flow4_inv, scale_factor=2, mode='bilinear', align_corners=True), out_conv2)
        concat3_inv = torch.cat((out_upconv3_inv, out_conv2, flow4_up_inv), 1)
        out_jconv3_inv = self.jconv3_inv(concat3_inv)
        flow3_inv = self.alpha * self.predict_flow3_inv(out_jconv3_inv)

        out_upconv2_inv = crop_like(self.upconv2_inv(out_jconv3_inv), out_conv1)
        flow3_up_inv = crop_like(F.interpolate(flow3_inv, scale_factor=2, mode='bilinear', align_corners=True), out_conv1)
        concat2_inv = torch.cat((out_upconv2_inv, out_conv1, flow3_up_inv), 1)
        out_jconv2_inv = self.jconv2_inv(concat2_inv)
        flow2_inv = self.alpha * self.predict_flow2_inv(out_jconv2_inv)

        out_upconv1_inv = crop_like(self.upconv1_inv(out_jconv2_inv), x1)
        flow2_up_inv = crop_like(F.interpolate(flow2_inv, scale_factor=2, mode='bilinear', align_corners=True), x1)
        concat1_inv = torch.cat((out_upconv1_inv, flow2_up_inv), 1)
        out_jconv1_inv = self.jconv1_inv(concat1_inv)
        flow1_inv = self.alpha * self.predict_flow1_inv(out_jconv1_inv)

        """ Forward branch """
        out_upconv3_fwd = crop_like(self.upconv3_fwd(out_jconv4), out_conv2)
        flow4_up_fwd = crop_like(F.interpolate(flow4_fwd, scale_factor=2, mode='bilinear', align_corners=True), out_conv2)
        concat3_fwd = torch.cat((out_upconv3_fwd, out_conv2, flow4_up_fwd), 1)
        out_jconv3_fwd = self.jconv3_fwd(concat3_fwd)
        flow3_fwd = self.alpha * self.predict_flow3_fwd(out_jconv3_fwd)

        out_upconv2_fwd = crop_like(self.upconv2_fwd(out_jconv3_fwd), out_conv1)
        flow3_up_fwd = crop_like(F.interpolate(flow3_fwd, scale_factor=2, mode='bilinear', align_corners=True), out_conv1)
        concat2_fwd = torch.cat((out_upconv2_fwd, out_conv1, flow3_up_fwd), 1)
        out_jconv2_fwd = self.jconv2_fwd(concat2_fwd)
        flow2_fwd = self.alpha * self.predict_flow2_fwd(out_jconv2_fwd)

        out_upconv1_fwd = crop_like(self.upconv1_fwd(out_jconv2_fwd), x1)
        flow2_up_fwd = crop_like(F.interpolate(flow2_fwd, scale_factor=2, mode='bilinear', align_corners=True), x1)
        concat1_fwd = torch.cat((out_upconv1_fwd, flow2_up_fwd), 1)
        out_jconv1_fwd = self.jconv1_fwd(concat1_fwd)
        flow1_fwd = self.alpha * self.predict_flow1_fwd(out_jconv1_fwd)

        """ Output concat """
        flow4 = torch.cat((flow4_inv, flow4_fwd), 1)    # torch.Size([2, 4, 56, 88])
        flow3 = torch.cat((flow3_inv, flow3_fwd), 1)    # torch.Size([2, 4, 112, 176])
        flow2 = torch.cat((flow2_inv, flow2_fwd), 1)    # torch.Size([2, 4, 224, 352])
        flow1 = torch.cat((flow1_inv, flow1_fwd), 1)    # torch.Size([2, 4, 448, 704])

        if self.training:
            return flow1, flow2, flow3, flow4
        else:
            return flow1
