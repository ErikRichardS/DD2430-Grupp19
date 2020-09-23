import torch
from torch.utils.data import TensorDataset, DataLoader
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as F

import os
from PIL import Image
import random


img_size = 256



#transform = transforms.Compose([
#	transforms.Resize((img_size,img_size)),
#	transforms.ToTensor()
#])


def get_training_data():


	dataset = ImageDataset("Data/img_train_shape", "Data/img_train_skeleton")

	return dataset




def get_random_transform(p_padding=0.5, p_vflip = 0.5, p_hflip=0.3):
	transform_list = []

	padding = random.uniform(0,1) < p_padding
	vflip = random.uniform(0,1) < p_vflip
	hflip = random.uniform(0,1) < p_hflip

	def rand_int():
		return random.randint(1,20)

	#transform_list.append( transforms.Resize((img_size,img_size)) )

	if padding:
		padding_tuple = ( rand_int(), rand_int(), rand_int(), rand_int() )
		transform_list.append( transforms.Pad(padding_tuple) )
		transform_list.append( transforms.CenterCrop(img_size) )

	if vflip:
		transform_list.append( transforms.functional.vflip )

	if hflip:
		transform_list.append( transforms.functional.hflip )

	
	transform_list.append(transforms.ToTensor())

	return transforms.Compose(transform_list)



# Dataset for handling the image input and output. 
# Takes the directory to the input and output files. 
# Requires the input and output files to have the same names.
class ImageDataset(torch.utils.data.Dataset):
	def __init__(self, train_directory, label_directory):
		self.trn_dir = train_directory
		self.lbl_dir = label_directory
		self.file_list = os.listdir( train_directory)
		self.file_list.remove(".DS_Store")


	def __getitem__(self, idx):
		transform = get_random_transform()

		# Load the img and turn it into a Torch tensor matrix
		link = self.trn_dir+"/"+self.file_list[idx]
		data = transform( Image.open(link) )


		# Create label
		link = self.lbl_dir+"/"+self.file_list[idx]
		label = transform( Image.open(link) )

		return (data, label)


	def __len__(self):
		return len(self.file_list)