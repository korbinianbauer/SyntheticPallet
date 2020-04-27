import glob
import random
import shutil
import os
import directories

train_val_split = 0
print('Set will be divided into ' + str(100 - train_val_split) + '% training and ' + str(train_val_split) + '% validation')


# Read in all input images
file_list_dir = glob.glob(directories.image_dir_input + '*.png')
file_list = []
for name in file_list_dir:
    data = name.replace(directories.image_dir_input, "")
    data = data.split('.png')[0]
    file_list.append(data)

n = len(file_list)
split = int(round(n * train_val_split / 100))
print('Number of images in the data set = ' + str(n))
print ('Number of resulting images for validation = ' + str(split))

val_list = random.sample(file_list, split)

# Make sure output folders exist
os.makedirs(os.path.dirname(directories.image_dir_val))
os.makedirs(os.path.dirname(directories.label_dir_val))
os.makedirs(os.path.dirname(directories.image_dir_train))
os.makedirs(os.path.dirname(directories.label_dir_train))

# Copy images and labels to output folders
for element in file_list:
    if element in val_list:
        shutil.copy(directories.image_dir_input + element + '.png', directories.image_dir_val)
        shutil.copy(directories.label_dir_input + element + '.txt', directories.label_dir_val)
    else:
        shutil.copy(directories.image_dir_input + element + '.png', directories.image_dir_train)
        shutil.copy(directories.label_dir_input + element + '.txt', directories.label_dir_train)

#print('Data set successfully split into training and validation sets !')
