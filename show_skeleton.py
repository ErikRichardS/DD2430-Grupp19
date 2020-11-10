import torch 
import torchvision.transforms as transforms

from PIL import Image

from ann import *
from eval import formalize_skeleton


image_to_tensor = transforms.ToTensor()
tensor_to_image = transforms.ToPILImage()


# Takes a pil image and a network to show the image's skeleton
def show_image_skeleton(img, net):
	data = torch.unsqueeze( image_to_tensor(img), 0 ).cuda()

	output = torch.squeeze( net(data).detach().cpu() )

	output_image = formalize_skeleton( tensor_to_image(output) )


	output_image.show()




#img_bat = Image.open("Data/img_train_shape/bat-1.png")

#ann = torch.load("skeleton_net.pt")


#show_image_skeleton(img_bat, ann)


