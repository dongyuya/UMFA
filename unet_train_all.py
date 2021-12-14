import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch.autograd import Variable
from tqdm import tqdm, trange
from function import  adaptive_instance_normalization

import torch.optim as optim
import random
import scipy.io as scio


import numpy as np
import time

from data_loader import list_images
import data_loader


from unet import Dense_net

# ------- 1. define loss function --------
from function import perceptual_loss, content_loss
import pytorch_msssim
mse_loss = torch.nn.MSELoss()
ssim_loss = pytorch_msssim.msssim

# ------- 2. set the directory of training dataset --------

# model_name = 'u2netp' #'u2netp'

# data_dir = './train_data/'
con_dir = "/data/Disk_B/MSCOCO2014/train2014/"
sty_dir = "/data/Disk_B/MSCOCO2014/train2014/"




epoch_num = 5
batch_size_train = 4
train_num = 40000

# tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*' + image_ext)
con_list = list_images(con_dir)
sty_list = list_images(sty_dir)

con_list = con_list[:train_num]
sty_list = sty_list[train_num:2*train_num]
random.shuffle(con_list)
random.shuffle(sty_list)


#args
class args():
	HEIGHT = 256
	WIDTH = 256
	log_interval = 5
	save_model_dir = "models"
	save_loss_dir = "models/loss"
	alpha = 0.5 #weight of style
	beta = (1-alpha)*0.5 #weight of content
	gama = (1-alpha)*0.5 #weight of style SSIM
	cuda = 1
	# model_path = "./models/Epoch_0_iters_250_Fri_May_22_15_45_17_2020_.model"
	# resume = "./models/Epoch_0_iters_1000.model"
	resume = None

# salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=1)

# ------- 3. define model --------
#
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
net = Dense_net()
net.apply(weights_init)

if args.resume is not None:
		print('Resuming, initializing using weight from {}.'.format(args.resume))
		net.load_state_dict(torch.load(args.resume))
print(net)

if torch.cuda.is_available():
	net.cuda()

# ------- 4. define optimizer --------
# print("---define optimizer...")
optimizer = optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=40,
# 														   verbose=True, threshold=0.0001, threshold_mode='rel',
# 														   cooldown=0, min_lr=0, eps=1e-12)

# ------- 5. training process --------
# print("---start training...")



tbar = trange(epoch_num)
print('Start training.....')

# creating save path
temp_path_model = os.path.join(args.save_model_dir)
if os.path.exists(temp_path_model) is False:
		os.mkdir(temp_path_model)

temp_path_loss = os.path.join(args.save_loss_dir)
if os.path.exists(temp_path_loss) is False:
		os.mkdir(temp_path_loss)

Loss_con = []
Loss_sty = []
Loss_ssim = []
Loss_all = []

all_con_loss = 0.
all_ssim_loss = 0
all_sty_loss = 0.


for e in tbar:
	print('Epoch %d.....' % e)
	# load training database
	con_set_ir, batches = data_loader.load_dataset(con_list, batch_size_train)
	sty_set_ir, sty_batches = data_loader.load_dataset(sty_list, batch_size_train)
	net.train()
	count = 0

	for batch in range(batches):
		image_paths = con_set_ir[batch * batch_size_train:(batch * batch_size_train + batch_size_train)]
		img = data_loader.get_train_images_auto(image_paths, height=args.HEIGHT, width=args.WIDTH)
		# style
		sty_image_paths = sty_set_ir[batch * batch_size_train:(batch * batch_size_train + batch_size_train)]
		sty_img = data_loader.get_train_images_auto(sty_image_paths, height=args.HEIGHT, width=args.WIDTH)
		count += 1
		optimizer.zero_grad()
		
		
		img = Variable(img, requires_grad=False)
		sty_img = Variable(sty_img, requires_grad=False)
		
		if args.cuda:
			img = img.cuda()
			sty_img = sty_img.cuda()
		#trans
		
		outputs = net(img, sty_img)
		
		# resolution loss
		x = Variable(img.data.clone(), requires_grad=False)
		sty = Variable(sty_img.data.clone(), requires_grad=False)
		
		con_loss_value = 0.
		ssim_loss_value = 0.
		sty_loss_value = 0.
	
		con_loss_temp = content_loss(outputs, x)
		ssim_loss_temp = ssim_loss(outputs, x, normalize=True)
		sty_loss_temp = perceptual_loss(outputs, sty)
		
		ssim_loss_value += (1 - ssim_loss_temp)
		con_loss_value += con_loss_temp
		sty_loss_value += sty_loss_temp
		
		ssim_loss_value /= len(outputs)
		con_loss_value /= len(outputs) * 3 * 256 * 256
		sty_loss_value /= len(outputs) * 3 * 256 * 256
		
		total_loss = con_loss_value*args.beta + ssim_loss_value*args.gama + sty_loss_value*args.alpha
		total_loss.backward()
		optimizer.step()
		# scheduler.step(total_loss.item())
		
		all_ssim_loss += ssim_loss_value.item()*args.gama
		all_con_loss += con_loss_value.item()*args.beta
		all_sty_loss += sty_loss_value.item()*args.alpha
		
		if (batch + 1) % args.log_interval == 0:
			mesg = "{}\tEpoch {}:\t[{}/{}]\t con loss: {:.6f}\t ssim loss: {:.6f}\t sty loss: {:.6f}\t total loss: {:.6f}\t ".format(
				time.ctime(), e + 1, count, batches,
				# optimizer.param_groups[0]['lr'],
							  all_con_loss / args.log_interval,
							  all_ssim_loss / args.log_interval,
							  all_sty_loss / args.log_interval,
							  (all_con_loss + all_ssim_loss + all_sty_loss ) / args.log_interval, )
			tbar.set_description(mesg)
			Loss_con.append(all_con_loss / args.log_interval)
			Loss_ssim.append(all_ssim_loss / args.log_interval)
			Loss_sty.append(all_sty_loss / args.log_interval)
			Loss_all.append((all_con_loss + all_ssim_loss + all_sty_loss) / args.log_interval)
			
			all_con_loss = 0.
			all_sty_loss = 0.
			all_ssim_loss = 0
		
		if (batch + 1) % (1000) == 0:
			# save model
			net.eval()
			net.cpu()
			save_model_filename = "Epoch_" + str(e) + "_iters_" + str(count) + ".model"
			save_model_path = os.path.join(args.save_model_dir, save_model_filename)
			torch.save(net.state_dict(), save_model_path)
			# save loss data
			# con loss
			loss_data_con = np.array(Loss_con)
			loss_filename_path = "loss_con_epoch_" + str(e) + "_iters_" + str(count) + "_" + ".mat"
			save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
			scio.savemat(save_loss_path, {'loss_con': loss_data_con})
			# ssim loss
			loss_data_ssim = np.array(Loss_ssim)
			loss_filename_path = "loss_ssim_epoch_" + str(e) + "_iters_" + str(count) + "_" + ".mat"
			save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
			scio.savemat(save_loss_path, {'loss_ssim': loss_data_ssim})
			# sty loss
			loss_data_sty = np.array(Loss_sty)
			loss_filename_path = "loss_sty_epoch_" + str(e) + "_iters_" + str(count) + "_" + ".mat"
			save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
			scio.savemat(save_loss_path, {'loss_sty': loss_data_sty})
			# all loss
			loss_data_total = np.array(Loss_all)
			loss_filename_path = "loss_total_epoch_" + str(e) + "_iters_" + str(count) + "_" + ".mat"
			save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
			scio.savemat(save_loss_path, {'loss_all': loss_data_total})
			#
			net.train()
			net.cuda()
			tbar.set_description("\nCheckpoint, trained model saved at", save_model_path)

# con loss
loss_data_con = np.array(Loss_con)
loss_filename_path =  "Final_loss_con_epoch_" + str(epoch_num) + "_" + ".mat"
save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
scio.savemat(save_loss_path, {'loss_con': loss_data_con})

# ssim loss
loss_data_ssim = np.array(Loss_ssim)
loss_filename_path =  "Final_loss_ssim_epoch_" + str(epoch_num) + "_" + ".mat"
save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
scio.savemat(save_loss_path, {'loss_ssim': loss_data_ssim})

# sty loss
loss_data_sty = np.array(Loss_sty)
loss_filename_path =  "Final_loss_sty_epoch_" + str(epoch_num) + "_" +  ".mat"
save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
scio.savemat(save_loss_path, {'loss_sty': loss_data_sty})

# all loss
loss_data_total = np.array(Loss_all)
loss_filename_path =  "Final_loss_all_epoch_" + str(epoch_num) + "_" + ".mat"
save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
scio.savemat(save_loss_path, {'loss_total': loss_data_total})

# save model
net.eval()
net.cpu()
save_model_filename = "Final_epoch_" + str(epoch_num)  + ".model"
save_model_path = os.path.join(args.save_model_dir, save_model_filename)
torch.save(net.state_dict(), save_model_path)

print("\nDone, trained model saved at", save_model_path)

if __name__ == "__main__":
	main()
