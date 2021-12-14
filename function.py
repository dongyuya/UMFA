import torch


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert (feat.size()[0] == 3)
    assert (isinstance(feat, torch.FloatTensor))
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std


import torch
import torch.nn as nn
from torchvision import models


class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()
        
        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])
        
        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        # out = h_relu_1_2
        return out
   
def totoal_variation_loss(output):
	diff_i = torch.sum(torch.abs(output[:, :, :, 1:] - output[:, :, :, :-1]))
	diff_j = torch.sum(torch.abs(output[:, :, 1:, :] - output[:, :, :-1, :]))
	tv_loss = (diff_i + diff_j)
	return tv_loss

def gram(x):
    (bs, ch, h, w) = x.size()
    f = x.view(bs, ch, w*h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (ch * h * w)
    return G
 
def perceptual_loss(output, sty):
	mse_loss = torch.nn.MSELoss()
	vgg = Vgg16().cuda()
	style_features = vgg(sty)
	out_features = vgg(output)
	style_gram = [gram(sty_map) for sty_map in style_features]
	out_gram = [gram(out_map) for out_map in out_features]
	p_loss = 0
	for j in range(2):
		p_loss += mse_loss(out_gram[j], style_gram[j])
	# p_loss = mse_loss(gram(out_features),gram(style_features))
	return p_loss

def content_loss(output, x):
	mse_loss = torch.nn.MSELoss()
	vgg = Vgg16().cuda()
	con_features = vgg(x)[1]
	out_features = vgg(output)[1]
	c_loss = mse_loss(out_features, con_features)
	return c_loss