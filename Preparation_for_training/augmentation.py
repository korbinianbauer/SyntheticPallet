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

      ###### Data Augmentation Call 2 ############
      image_aug2, bbs_aug2 = iaa.GaussianBlur(sigma=(0.3, 2.0))(image=np_im, bounding_boxes=bbs)
      da2_im = Image.fromarray(image_aug2)
      width_dim, height_dim = da2_im.size
      da2_im.save(folder_dir_images + "/da_2_" + fname.split('.png')[0] + ".png")
      with open(folder_dir_label + "/da_2_" + fname.split('.png')[0] + ".txt", "w") as f:
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
                  cropped_bbs += 1

               xmid = float(float(xmax - xmin) / 2 + xmin) / width_dim
               ymid = float(float(ymax - ymin) / 2 + ymin) / height_dim
               xrel = float(float(xmax - xmin) / width_dim)
               yrel = float(float(ymax - ymin) / height_dim)


               class_name = dictionary_class[nr]

               f.write(str(class_name) + " " + str(round(xmid,6)) + " " + str(round(ymid,6)) + " " + str(round(xrel,6)) + " " + str(round(yrel,6)) + "\n")


      f.close()
      ####### Data Augmentation Call 4 ############
      image_aug4, bbs_aug4 = iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))(image=np_im, bounding_boxes=bbs)
      da4_im = Image.fromarray(image_aug4)
      width_dim, height_dim = da4_im.size
      da4_im.save(folder_dir_images + "/da_4_" + fname.split('.png')[0] + ".png")
      with open(folder_dir_label + "/da_4_" + fname.split('.png')[0] + ".txt", "w") as f:
         for nr in range(0, len(bbs_aug4.bounding_boxes)):
               xmin = bbs_aug4.bounding_boxes[nr].x1
               xmax = bbs_aug4.bounding_boxes[nr].x2
               ymin = bbs_aug4.bounding_boxes[nr].y1
               ymax = bbs_aug4.bounding_boxes[nr].y2

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
                  cropped_bbs += 1

               xmid = float(float(xmax - xmin) / 2 + xmin) / width_dim
               ymid = float(float(ymax - ymin) / 2 + ymin) / height_dim
               xrel = float(float(xmax - xmin) / width_dim)
               yrel = float(float(ymax - ymin) / height_dim)

               class_name = dictionary_class[nr]

               f.write(str(class_name) + " " + str(round(xmid,6)) + " " + str(round(ymid,6)) + " " + str(round(xrel,6)) + " " + str(round(yrel,6)) + "\n")


      f.close()
      ####### Data Augmentation Call 9 ############
      image_aug9, bbs_aug9 = iaa.AddToHueAndSaturation((-40, 40))(image=np_im, bounding_boxes=bbs)
      da9_im = Image.fromarray(image_aug9)
      width_dim, height_dim = da9_im.size
      da9_im.save(folder_dir_images + "/da_9_" + fname.split('.png')[0] + ".png")
      with open(folder_dir_label + "/da_9_" + fname.split('.png')[0] + ".txt", "w") as f:
         for nr in range(0, len(bbs_aug9.bounding_boxes)):
               xmin = bbs_aug9.bounding_boxes[nr].x1
               xmax = bbs_aug9.bounding_boxes[nr].x2
               ymin = bbs_aug9.bounding_boxes[nr].y1
               ymax = bbs_aug9.bounding_boxes[nr].y2

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
                  cropped_bbs += 1

               xmid = float(float(xmax - xmin) / 2 + xmin) / width_dim
               ymid = float(float(ymax - ymin) / 2 + ymin) / height_dim
               xrel = float(float(xmax - xmin) / width_dim)
               yrel = float(float(ymax - ymin) / height_dim)

               class_name = dictionary_class[nr]

               f.write(str(class_name) + " " + str(round(xmid,6)) + " " + str(round(ymid,6)) + " " + str(round(xrel,6)) + " " + str(round(yrel,6)) + "\n")


      f.close()
      ####### Data Augmentation Call 10 ############
      image_aug10, bbs_aug10 = iaa.LinearContrast((0.6, 2.5), per_channel=0.5)(image=np_im, bounding_boxes=bbs)
      da10_im = Image.fromarray(image_aug10)
      width_dim, height_dim = da10_im.size
      da10_im.save(folder_dir_images + "/da_10_" + fname.split('.png')[0] + ".png")
      with open(folder_dir_label + "/da_10_" + fname.split('.png')[0] + ".txt", "w") as f:
         for nr in range(0, len(bbs_aug10.bounding_boxes)):
               xmin = bbs_aug10.bounding_boxes[nr].x1
               xmax = bbs_aug10.bounding_boxes[nr].x2
               ymin = bbs_aug10.bounding_boxes[nr].y1
               ymax = bbs_aug10.bounding_boxes[nr].y2

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
                  cropped_bbs += 1

               xmid = float(float(xmax - xmin) / 2 + xmin) / width_dim
               ymid = float(float(ymax - ymin) / 2 + ymin) / height_dim
               xrel = float(float(xmax - xmin) / width_dim)
               yrel = float(float(ymax - ymin) / height_dim)


               class_name = dictionary_class[nr]

               f.write(str(class_name) + " " + str(round(xmid,6)) + " " + str(round(ymid,6)) + " " + str(round(xrel,6)) + " " + str(round(yrel,6)) + "\n")


      f.close()
      ####### Data Augmentation Call 11 ############
      image_aug11, bbs_aug11 = iaa.Grayscale(alpha=(0.2, 1.0))(image=np_im, bounding_boxes=bbs)
      da11_im = Image.fromarray(image_aug11)
      width_dim, height_dim = da11_im.size
      da11_im.save(folder_dir_images + "/da_11_" + fname.split('.png')[0] + ".png")
      with open(folder_dir_label + "/da_11_" + fname.split('.png')[0] + ".txt", "w") as f:
         for nr in range(0, len(bbs_aug11.bounding_boxes)):
               xmin = bbs_aug11.bounding_boxes[nr].x1
               xmax = bbs_aug11.bounding_boxes[nr].x2
               ymin = bbs_aug11.bounding_boxes[nr].y1
               ymax = bbs_aug11.bounding_boxes[nr].y2

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
                  cropped_bbs += 1

               xmid = float(float(xmax - xmin) / 2 + xmin) / width_dim
               ymid = float(float(ymax - ymin) / 2 + ymin) / height_dim
               xrel = float(float(xmax - xmin) / width_dim)
               yrel = float(float(ymax - ymin) / height_dim)


               class_name = dictionary_class[nr]

               f.write(str(class_name) + " " + str(round(xmid,6)) + " " + str(round(ymid,6)) + " " + str(round(xrel,6)) + " " + str(round(yrel,6)) + "\n")


      f.close()

      ####### Data Augmentation Call 14 ############
      image_aug14, bbs_aug14 = iaa.AdditiveGaussianNoise(scale=(0.1 * 225), per_channel=False)(image=np_im, bounding_boxes=bbs)
      da14_im = Image.fromarray(image_aug14)
      width_dim, height_dim = da14_im.size
      da14_im.save(folder_dir_images + "/da_14_" + fname.split('.png')[0] + ".png")
      with open(folder_dir_label + "/da_14_" + fname.split('.png')[0] + ".txt", "w") as f:
         for nr in range(0, len(bbs_aug14.bounding_boxes)):
               xmin = bbs_aug14.bounding_boxes[nr].x1
               xmax = bbs_aug14.bounding_boxes[nr].x2
               ymin = bbs_aug14.bounding_boxes[nr].y1
               ymax = bbs_aug14.bounding_boxes[nr].y2

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
                  cropped_bbs += 1

               xmid = float(float(xmax - xmin) / 2 + xmin) / width_dim
               ymid = float(float(ymax - ymin) / 2 + ymin) / height_dim
               xrel = float(float(xmax - xmin) / width_dim)
               yrel = float(float(ymax - ymin) / height_dim)

               class_name = dictionary_class[nr]

               f.write(str(class_name) + " " + str(round(xmid,6)) + " " + str(round(ymid,6)) + " " + str(round(xrel,6)) + " " + str(round(yrel,6)) + "\n")


      f.close()

      ####### Data Augmentation Call 15 ############
      image_aug15, bbs_aug15 = iaa.Multiply((0.5, 1.5), per_channel=True)(image=np_im, bounding_boxes=bbs)
      da15_im = Image.fromarray(image_aug15)
      width_dim, height_dim = da15_im.size
      da15_im.save(folder_dir_images + "/da_15_" + fname.split('.png')[0] + ".png")
      with open(folder_dir_label + "/da_15_" + fname.split('.png')[0] + ".txt", "w") as f:
         for nr in range(0, len(bbs_aug15.bounding_boxes)):
               xmin = bbs_aug15.bounding_boxes[nr].x1
               xmax = bbs_aug15.bounding_boxes[nr].x2
               ymin = bbs_aug15.bounding_boxes[nr].y1
               ymax = bbs_aug15.bounding_boxes[nr].y2

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
                  cropped_bbs += 1

               xmid = float(float(xmax - xmin) / 2 + xmin) / width_dim
               ymid = float(float(ymax - ymin) / 2 + ymin) / height_dim
               xrel = float(float(xmax - xmin) / width_dim)
               yrel = float(float(ymax - ymin) / height_dim)

               class_name = dictionary_class[nr]

               f.write(str(class_name) + " " + str(round(xmid,6)) + " " + str(round(ymid,6)) + " " + str(round(xrel,6)) + " " + str(round(yrel,6)) + "\n")


      f.close()
      ####### Data Augmentation Call 16 ############
      image_aug16, bbs_aug16 = iaa.Multiply((0.25, 0.5), per_channel=True)(image=np_im, bounding_boxes=bbs)
      da16_im = Image.fromarray(image_aug16)
      width_dim, height_dim = da16_im.size
      da16_im.save(folder_dir_images + "/da_16_" + fname.split('.png')[0] + ".png")
      with open(folder_dir_label + "/da_16_" + fname.split('.png')[0] + ".txt", "w") as f:
         for nr in range(0, len(bbs_aug16.bounding_boxes)):
               xmin = bbs_aug16.bounding_boxes[nr].x1
               xmax = bbs_aug16.bounding_boxes[nr].x2
               ymin = bbs_aug16.bounding_boxes[nr].y1
               ymax = bbs_aug16.bounding_boxes[nr].y2

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
                  cropped_bbs += 1

               xmid = float(float(xmax - xmin) / 2 + xmin) / width_dim
               ymid = float(float(ymax - ymin) / 2 + ymin) / height_dim
               xrel = float(float(xmax - xmin) / width_dim)
               yrel = float(float(ymax - ymin) / height_dim)

               class_name = dictionary_class[nr]

               f.write(str(class_name) + " " + str(round(xmid,6)) + " " + str(round(ymid,6)) + " " + str(round(xrel,6)) + " " + str(round(yrel,6)) + "\n")


      f.close()
      ####### Data Augmentation Call 17 ############
      image_aug17, bbs_aug17 = iaa.Multiply((0.5, 0.75), per_channel=True)(image=np_im, bounding_boxes=bbs)
      da17_im = Image.fromarray(image_aug17)
      width_dim, height_dim = da17_im.size
      da17_im.save(folder_dir_images + "/da_17_" + fname.split('.png')[0] + ".png")
      with open(folder_dir_label + "/da_17_" + fname.split('.png')[0] + ".txt", "w") as f:
         for nr in range(0, len(bbs_aug17.bounding_boxes)):
               xmin = bbs_aug17.bounding_boxes[nr].x1
               xmax = bbs_aug17.bounding_boxes[nr].x2
               ymin = bbs_aug17.bounding_boxes[nr].y1
               ymax = bbs_aug17.bounding_boxes[nr].y2

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
                  cropped_bbs += 1

               xmid = float(float(xmax - xmin) / 2 + xmin) / width_dim
               ymid = float(float(ymax - ymin) / 2 + ymin) / height_dim
               xrel = float(float(xmax - xmin) / width_dim)
               yrel = float(float(ymax - ymin) / height_dim)

               class_name = dictionary_class[nr]

               f.write(str(class_name) + " " + str(round(xmid,6)) + " " + str(round(ymid,6)) + " " + str(round(xrel,6)) + " " + str(round(yrel,6)) + "\n")


      f.close()
      ####### Data Augmentation Call 18 ############
      image_aug18, bbs_aug18 = iaa.Multiply((0.75, 1.0), per_channel=True)(image=np_im, bounding_boxes=bbs)
      da18_im = Image.fromarray(image_aug18)
      width_dim, height_dim = da18_im.size
      da18_im.save(folder_dir_images + "/da_18_" + fname.split('.png')[0] + ".png")
      with open(folder_dir_label + "/da_18_" + fname.split('.png')[0] + ".txt", "w") as f:
         for nr in range(0, len(bbs_aug18.bounding_boxes)):
               xmin = bbs_aug18.bounding_boxes[nr].x1
               xmax = bbs_aug18.bounding_boxes[nr].x2
               ymin = bbs_aug18.bounding_boxes[nr].y1
               ymax = bbs_aug18.bounding_boxes[nr].y2

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
                  cropped_bbs += 1


               xmid = float(float(xmax - xmin) / 2 + xmin) / width_dim
               ymid = float(float(ymax - ymin) / 2 + ymin) / height_dim
               xrel = float(float(xmax - xmin) / width_dim)
               yrel = float(float(ymax - ymin) / height_dim)

               class_name = dictionary_class[nr]

               f.write(str(class_name) + " " + str(round(xmid,6)) + " " + str(round(ymid,6)) + " " + str(round(xrel,6)) + " " + str(round(yrel,6)) + "\n")


      f.close()
      ####### Data Augmentation Call 19 ############
      image_aug19, bbs_aug19 = iaa.Multiply((1.0, 1.25), per_channel=True)(image=np_im, bounding_boxes=bbs)
      da19_im = Image.fromarray(image_aug19)
      width_dim, height_dim = da19_im.size
      da19_im.save(folder_dir_images + "/da_19_" + fname.split('.png')[0] + ".png")
      with open(folder_dir_label + "/da_19_" + fname.split('.png')[0] + ".txt", "w") as f:
         for nr in range(0, len(bbs_aug19.bounding_boxes)):
               xmin = bbs_aug19.bounding_boxes[nr].x1
               xmax = bbs_aug19.bounding_boxes[nr].x2
               ymin = bbs_aug19.bounding_boxes[nr].y1
               ymax = bbs_aug19.bounding_boxes[nr].y2

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
                  cropped_bbs += 1

               xmid = float(float(xmax - xmin) / 2 + xmin) / width_dim
               ymid = float(float(ymax - ymin) / 2 + ymin) / height_dim
               xrel = float(float(xmax - xmin) / width_dim)
               yrel = float(float(ymax - ymin) / height_dim)


               class_name = dictionary_class[nr]

               f.write(str(class_name) + " " + str(round(xmid,6)) + " " + str(round(ymid,6)) + " " + str(round(xrel,6)) + " " + str(round(yrel,6)) + "\n")


      f.close()
      ####### Data Augmentation Call 20 ############
      image_aug20, bbs_aug20 = iaa.Multiply((1.25, 1.5), per_channel=True)(image=np_im, bounding_boxes=bbs)
      da20_im = Image.fromarray(image_aug20)
      width_dim, height_dim = da20_im.size
      da20_im.save(folder_dir_images + "/da_20_" + fname.split('.png')[0] + ".png")
      with open(folder_dir_label + "/da_20_" + fname.split('.png')[0] + ".txt", "w") as f:
         for nr in range(0, len(bbs_aug20.bounding_boxes)):
               xmin = bbs_aug20.bounding_boxes[nr].x1
               xmax = bbs_aug20.bounding_boxes[nr].x2
               ymin = bbs_aug20.bounding_boxes[nr].y1
               ymax = bbs_aug20.bounding_boxes[nr].y2

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
                  cropped_bbs += 1

               xmid = float(float(xmax - xmin) / 2 + xmin) / width_dim
               ymid = float(float(ymax - ymin) / 2 + ymin) / height_dim
               xrel = float(float(xmax - xmin) / width_dim)
               yrel = float(float(ymax - ymin) / height_dim)


               class_name = dictionary_class[nr]

               f.write(str(class_name) + " " + str(round(xmid,6)) + " " + str(round(ymid,6)) + " " + str(round(xrel,6)) + " " + str(round(yrel,6)) + "\n")


      f.close()


      ####### Data Augmentation Call custom ############
      image_aug_c, bbs_aug_c = iaa.AdditiveGaussianNoise(scale=(0.025 * 225), per_channel=False)(image=np_im, bounding_boxes=bbs)
      image_aug_c, bbs_aug_c = iaa.GaussianBlur(sigma=0.5)(image=image_aug_c, bounding_boxes=bbs_aug_c)
      da_c_im = Image.fromarray(image_aug_c)
      width_dim, height_dim = da_c_im.size
      da_c_im.save(folder_dir_images + "/da_c_" + fname.split('.png')[0] + ".png")
      with open(folder_dir_label + "/da_c_" + fname.split('.png')[0] + ".txt", "w") as f:
         for nr in range(0, len(bbs_aug_c.bounding_boxes)):
               xmin = bbs_aug_c.bounding_boxes[nr].x1
               xmax = bbs_aug_c.bounding_boxes[nr].x2
               ymin = bbs_aug_c.bounding_boxes[nr].y1
               ymax = bbs_aug_c.bounding_boxes[nr].y2

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
                  cropped_bbs += 1

               xmid = float(float(xmax - xmin) / 2 + xmin) / width_dim
               ymid = float(float(ymax - ymin) / 2 + ymin) / height_dim
               xrel = float(float(xmax - xmin) / width_dim)
               yrel = float(float(ymax - ymin) / height_dim)

               class_name = dictionary_class[nr]

               f.write(str(class_name) + " " + str(round(xmid,6)) + " " + str(round(ymid,6)) + " " + str(round(xrel,6)) + " " + str(round(yrel,6)) + "\n")


      f.close()
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