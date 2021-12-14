# test phase

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
from function import adaptive_instance_normalization

# from u2net_train import args
import data_loader

import numpy as np
import time

from unet import Dense_net


# normalize the predicted SOD probability map
def load_model(path):
	# en_net = encoder()
	# pre_dict = torch.load(en_path)
	# model_dict = en_net.state_dict()
	# pre_dict = {k: v for k, v in pre_dict.items() if k in model_dict}
	# model_dict.update(pre_dict)
	# en_net.load_state_dict(model_dict)
	
	net = Dense_net()
	net.load_state_dict(torch.load(path))
	
	para = sum([np.prod(list(p.size())) for p in net.parameters()])
	type_size = 4
	print('Model {} : params: {:4f}M'.format(net._get_name(), para * type_size / 1000 / 1000))
	
	net.eval()
	net.cuda()

	
	return net


def _generate_fusion_image(model, con, sty):
	
	
	out = model(con, sty)
	
	
	return out


def run_demo(model, con_path, sty_path, output_path_root, index):
	con_img = data_loader.get_test_images(con_path, height=None, width=None)
	sty_img = data_loader.get_test_images(sty_path, height=None, width=None)
	
	out = data_loader.get_image(con_path, height=None, width=None)
	# dim = img_ir.shape
	
	con_img = con_img.cuda()
	sty_img = sty_img.cuda()
	con_img = Variable(con_img, requires_grad=False)
	sty_img = Variable(sty_img, requires_grad=False)
	# dimension = con_img.size()
	
	img_fusion = _generate_fusion_image(model, con_img, sty_img)
	############################ multi outputs ##############################################
	file_name = 'fusion_' + str(index) + '.png'
	output_path = output_path_root + file_name
	if torch.cuda.is_available():
		img = img_fusion.cpu().clamp(0, 255).data[0].numpy()
	else:
		img = img_fusion.clamp(0, 255).data[0].numpy()
	img = img.transpose(1, 2, 0).astype('uint8')
	data_loader.save_images(output_path, img, out)
	
	print(output_path)


def main():
	con_path = "images/content/"
	sty_path = "images/style/"
	# network_type = 'densefuse'
	
	output_path = './outputs/'
	# strategy_type = strategy_type_list[0]
	
	if os.path.exists(output_path) is False:
		os.mkdir(output_path)
	
	in_c = 3
	out_c = in_c
	model_path = "./models/Epoch_1_iters_10000.model"
	
	with torch.no_grad():
		
		model = load_model(model_path)
		for i in range(5):
			index = i + 1
			infrared_path = con_path + 'in' + str(index) + '.png'
			visible_path = sty_path + 'in'+ str(index)  + '.png'
			start = time.time()
			run_demo(model, infrared_path, visible_path, output_path, index)
			end = time.time()
			print('time:', end - start, 'S')
	print('Done......')


if __name__ == "__main__":
	main()
