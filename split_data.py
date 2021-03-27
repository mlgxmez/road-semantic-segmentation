import argparse
from utils.preprocessing import randomSubsetFiles, splitMasksFolder, copyFiles

parser = argparse.ArgumentParser()
parser.add_argument('--images_folder')
parser.add_argument('--masks_folder')

args = parser.parse_args()

# Create two lists containing train and evals file sets
images_bucket1, images_bucket2 = randomSubsetFiles(args.images_folder, 0.2)

# Rearrange masks following filename pattern in masks_bucket1 and masks_bucket2
masks_bucket1, masks_bucket2 = splitMasksFolder(
    args.masks_folder,
    images_bucket1,
    images_bucket2)

# Copy data of train set
copyFiles(images_bucket1, directory='data/train', appendParentDir=2)
copyFiles(masks_bucket1, directory='data/train', appendParentDir=2)

# Copy data of eval set
copyFiles(images_bucket2, directory='data/eval', appendParentDir=2)
copyFiles(masks_bucket2, directory='data/eval', appendParentDir=2)
