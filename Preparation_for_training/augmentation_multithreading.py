from __future__ import print_function
import sys
import time
import numpy as np
import imgaug.augmenters as iaa
from PIL import Image
import os
import glob
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import directories
import threading


image_dirs = [directories.image_dir_train]
label_dirs = [directories.label_dir_train]

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def aug_aftermath(aug_prefix, image_aug2, bbs_aug2, fname, dictionary_class, bbs):
   da2_im = Image.fromarray(image_aug2)
   width_dim, height_dim = da2_im.size
   da2_im.save(folder_dir_images + "/" + aug_prefix + fname.split('.png')[0] + ".png")
   with open(folder_dir_label + "/" + aug_prefix + fname.split('.png')[0] + ".txt", "w") as f:
      for nr in range(0, len(bbs_aug2.bounding_boxes)):
            xmin = bbs_aug2.bounding_boxes[nr].x1
            xmax = bbs_aug2.bounding_boxes[nr].x2
            ymin = bbs_aug2.bounding_boxes[nr].y1
            ymax = bbs_aug2.bounding_boxes[nr].y2


            had_to_crop = False
            if (xmin < 0):
               had_to_crop = True
               xmin = 0
            if (ymin < 0):
               had_to_crop = True
               ymin = 0
            if (xmax > width_dim):
               had_to_crop = True
               xmax = width_dim
            if (ymax > height_dim):
               had_to_crop = True
               ymax = height_dim

            if had_to_crop:
               #cropped_bbs += 1
               pass

            xmid = float(float(xmax - xmin) / 2 + xmin) / width_dim
            ymid = float(float(ymax - ymin) / 2 + ymin) / height_dim
            xrel = float(float(xmax - xmin) / width_dim)
            yrel = float(float(ymax - ymin) / height_dim)


            class_name = dictionary_class[nr]

            f.write(str(class_name) + " " + str(round(xmid,6)) + " " + str(round(ymid,6)) + " " + str(round(xrel,6)) + " " + str(round(yrel,6)) + "\n")


   f.close()

def aug2(fname, dictionary_class, np_im, bbs):
   ###### Data Augmentation Call 2 ############
      image_aug, bbs_aug = iaa.GaussianBlur(sigma=(0.3, 2.0))(image=np_im, bounding_boxes=bbs)
      aug_aftermath('da_2_', image_aug, bbs_aug, fname, dictionary_class, bbs)
      
def aug4(fname, dictionary_class, np_im, bbs):
      image_aug, bbs_aug = iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))(image=np_im, bounding_boxes=bbs)
      aug_aftermath('da_4_', image_aug, bbs_aug, fname, dictionary_class, bbs)

def aug9(fname, dictionary_class, np_im, bbs):
      image_aug, bbs_aug = iaa.AddToHueAndSaturation((-40, 40))(image=np_im, bounding_boxes=bbs)
      aug_aftermath('da_9_', image_aug, bbs_aug, fname, dictionary_class, bbs)

def aug10(fname, dictionary_class, np_im, bbs):
      image_aug, bbs_aug = iaa.LinearContrast((0.6, 2.5), per_channel=0.5)(image=np_im, bounding_boxes=bbs)
      aug_aftermath('da_10_', image_aug, bbs_aug, fname, dictionary_class, bbs)

def aug11(fname, dictionary_class, np_im, bbs):
      image_aug, bbs_aug = iaa.Grayscale(alpha=(0.2, 1.0))(image=np_im, bounding_boxes=bbs)
      aug_aftermath('da_11_', image_aug, bbs_aug, fname, dictionary_class, bbs)

def aug14(fname, dictionary_class, np_im, bbs):
      image_aug, bbs_aug = iaa.AdditiveGaussianNoise(scale=(0.1 * 225), per_channel=False)(image=np_im, bounding_boxes=bbs)
      aug_aftermath('da_14_', image_aug, bbs_aug, fname, dictionary_class, bbs)

def aug15(fname, dictionary_class, np_im, bbs):
      image_aug, bbs_aug = iaa.Multiply((0.5, 1.5), per_channel=True)(image=np_im, bounding_boxes=bbs)
      aug_aftermath('da_15_', image_aug, bbs_aug, fname, dictionary_class, bbs)

def aug16(fname, dictionary_class, np_im, bbs):
      image_aug, bbs_aug = iaa.Multiply((0.25, 0.5), per_channel=True)(image=np_im, bounding_boxes=bbs)
      aug_aftermath('da_16_', image_aug, bbs_aug, fname, dictionary_class, bbs)

def aug17(fname, dictionary_class, np_im, bbs):
      image_aug, bbs_aug = iaa.Multiply((0.5, 0.75), per_channel=True)(image=np_im, bounding_boxes=bbs)
      aug_aftermath('da_17_', image_aug, bbs_aug, fname, dictionary_class, bbs)

def aug18(fname, dictionary_class, np_im, bbs):
      image_aug, bbs_aug = iaa.Multiply((0.75, 1.0), per_channel=True)(image=np_im, bounding_boxes=bbs)
      aug_aftermath('da_18_', image_aug, bbs_aug, fname, dictionary_class, bbs)

def aug19(fname, dictionary_class, np_im, bbs):
      image_aug, bbs_aug = iaa.Multiply((1.0, 1.25), per_channel=True)(image=np_im, bounding_boxes=bbs)
      aug_aftermath('da_19_', image_aug, bbs_aug, fname, dictionary_class, bbs)

def aug20(fname, dictionary_class, np_im, bbs):
      image_aug, bbs_aug = iaa.Multiply((1.25, 1.5), per_channel=True)(image=np_im, bounding_boxes=bbs)
      aug_aftermath('da_20_', image_aug, bbs_aug, fname, dictionary_class, bbs)

def aug_custom(fname, dictionary_class, np_im, bbs):
      image_aug_c, bbs_aug_c = iaa.AdditiveGaussianNoise(scale=(0.025 * 225), per_channel=False)(image=np_im, bounding_boxes=bbs)
      image_aug, bbs_aug = iaa.GaussianBlur(sigma=0.5)(image=image_aug_c, bounding_boxes=bbs_aug_c)
      aug_aftermath('da_c_', image_aug, bbs_aug, fname, dictionary_class, bbs)



for dataset_index in range(len(image_dirs)):

   folder_dir_images = image_dirs[dataset_index]
   folder_dir_label = label_dirs[dataset_index]
   filelist = glob.glob(folder_dir_images+"*.png")
   #print(filelist)
   print('Augmenting {} samples for dataset #{}'.format(len(filelist), dataset_index))

   filecount = len(filelist)
   files_done = 0
   start_time = time.time()
   cropped_bbs = 0

   for fname in filelist:
      fname = fname.replace(folder_dir_images,"")
      data = open(folder_dir_label + fname.split('.png')[0] + '.txt', "r")
      dictionary_class = {}
      dictionary_points = {}

      im = Image.open(folder_dir_images + fname)
      width_dim, height_dim = im.size

      for index, line in enumerate(data):
         class_name = str(line.split()[0])
         xmid = float(line.split()[1])
         ymid = float(line.split()[2])
         xrel = float(float(line.split()[3]) / 2)
         yrel = float(float(line.split()[4]) / 2)

         xmin = int(round(xmid - xrel, 3) * width_dim)
         ymin = int(round(ymid - yrel, 3) * height_dim)
         xmax = int(round(xmid + xrel, 3) * width_dim)
         ymax = int(round(ymid + yrel, 3) * height_dim)

         dictionary_class[index] = class_name
         dictionary_points[index] = [xmin, xmax, ymin, ymax]

      np_im = np.array(im)

      bbx_list = []
      for bbox_nr in range(0, len(dictionary_points)):
         bbx_list.append(BoundingBox(x1=dictionary_points[bbox_nr][0], x2=dictionary_points[bbox_nr][1],
                                       y1=dictionary_points[bbox_nr][2], y2=dictionary_points[bbox_nr][3]))

      bbs = BoundingBoxesOnImage(bbx_list, shape=np_im.shape)

      #print('bbx= ' + str(bbx_list) + '/n')
      #print('bbs= ' + str(bbs) + '/n')

      aug2_thread = threading.Thread(target=aug2, args=(fname, dictionary_class, np_im, bbs))
      aug2_thread.start()

      aug4_thread = threading.Thread(target=aug4, args=(fname, dictionary_class, np_im, bbs))
      aug4_thread.start()

      aug9_thread = threading.Thread(target=aug9, args=(fname, dictionary_class, np_im, bbs))
      aug9_thread.start()

      aug10_thread = threading.Thread(target=aug10, args=(fname, dictionary_class, np_im, bbs))
      aug10_thread.start()

      aug11_thread = threading.Thread(target=aug11, args=(fname, dictionary_class, np_im, bbs))
      aug11_thread.start()

      aug14_thread = threading.Thread(target=aug14, args=(fname, dictionary_class, np_im, bbs))
      aug14_thread.start()

      aug15_thread = threading.Thread(target=aug15, args=(fname, dictionary_class, np_im, bbs))
      aug15_thread.start()

      aug16_thread = threading.Thread(target=aug16, args=(fname, dictionary_class, np_im, bbs))
      aug16_thread.start()

      aug17_thread = threading.Thread(target=aug17, args=(fname, dictionary_class, np_im, bbs))
      aug17_thread.start()

      aug18_thread = threading.Thread(target=aug18, args=(fname, dictionary_class, np_im, bbs))
      aug18_thread.start()

      aug19_thread = threading.Thread(target=aug19, args=(fname, dictionary_class, np_im, bbs))
      aug19_thread.start()

      aug20_thread = threading.Thread(target=aug20, args=(fname, dictionary_class, np_im, bbs))
      aug20_thread.start()

      aug_custom_thread = threading.Thread(target=aug_custom, args=(fname, dictionary_class, np_im, bbs))
      aug_custom_thread.start()

      aug2_thread.join()
      aug4_thread.join()
      aug9_thread.join()
      aug10_thread.join()
      aug11_thread.join()
      aug14_thread.join()
      aug15_thread.join()
      aug16_thread.join()
      aug17_thread.join()
      aug18_thread.join()
      aug19_thread.join()
      aug20_thread.join()
      aug_custom_thread.join()
      #aug2(fname, dictionary_class, np_im, bbs)


      files_done += 1

      files_left = filecount - files_done
      time_per_file = (time.time() - start_time)/files_done

      time_left = int(files_left * time_per_file)
      time_left_string = time.strftime('%H:%M:%S', time.gmtime(time_left))
      
      #print('Image processed')
      print(bcolors.OKBLUE + "Augmented {}/{} samples. ETA: {}s                                 ".format(files_done, filecount, time_left_string) + bcolors.ENDC, end = '\r')
      sys.stdout.flush()

   print('Augmentation finished for dataset #{}                                 '.format(dataset_index))
   if cropped_bbs > 0:
      print (bcolors.WARNING + 'Had to crop {} BBs, because they were out of the image border'.format(cropped_bbs) + bcolors.ENDC)
print('Augmentation finished.')