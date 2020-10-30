import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.morphology import skeletonize, medial_axis
import os

# image data directory (pre-segmented)
img_dir = ""

# Directory paths to save altered data.
""" 
    Example:
        dataset_path = "Data/Stanford_Dogs"
        train_path = "Data/Stanford_Dogs/train_imgs"
        test_skeletonize_path = "Data/Stanford_Dogs/test_skeletonze_imgs"
        test_medial_axis_path = "Data/Stanford_Dogs/test_medial_imgs"
"""
dataset_path = ""
train_path = ""
test_skeletonize_path = ""
test_medial_axis_path = ""


def create_skeletons(images):
    try:
        # Create target directories
        os.mkdir(dataset_path)
        os.mkdir(train_path)
        os.mkdir(test_skeletonize_path)
        os.mkdir(test_medial_axis_path)
    except FileExistsError:
        print("Directory already exists")

    for i in range(len(images)):
        res = cv2.resize(images[i], dsize=(256, 256),
                         interpolation=cv2.INTER_CUBIC)
        plt.imsave(fname=train_path + "/train_" +
                   str(i) + ".png", arr=res, cmap="gray")
        res_medial = medial_axis(res[:, :, 0])
        plt.imsave(fname=test_medial_axis_path + "/medial_" +
                   str(i) + ".png", arr=res_medial, cmap="gray")

        res_skeleton = skeletonize(res[:, :, 0])
        plt.imsave(fname=test_skeletonize_path + "/skeletonize_" +
                   str(i) + ".png", arr=res_skeleton, cmap="gray")


def main():

    # get list of all samples
    samples = []
    for file in os.listdir(img_dir):
        if file != ".DS_Store":
            samples.append(plt.imread(os.path.join(img_dir, file)))

    # Create skeletons (medial axis & skeletonize) and save in folders
    create_skeletons(samples)

    print("Dataset created.")


if __name__ == "__main__":
    main()
