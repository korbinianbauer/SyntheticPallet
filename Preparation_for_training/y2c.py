import os, sys
import os
import codecs
import io
from PIL import Image
import directories


ourDict = {0: 'KLT6410', 1: 'KLT6147', 2: 'KLT6412', 3: 'KLT2589', 4: 'label' }


def yolo2csv(labelDirectory, imageDirectory, imageEnding, labelFileDest, associationFileDest):
    # list of the names for the images

    names_list = [os.path.splitext(file)[0] for file in os.listdir(labelDirectory) if file.endswith('.txt')]
    pics_list = [imageDirectory+n+'.png' for n in names_list]
    labels_list = [labelDirectory+n+'.txt' for n in names_list]

    overallWriteString = ""

    print("{} labels found".format(len(labels_list)))

    for i in range(0,len(labels_list)):

        #get to the label
        with open(labels_list[i], 'r') as f:

            imageContent = f.read().split('\n')[0:-1]

            listOfPictureContents = [ic.split() for ic in imageContent]
            #print("\n"+str(listOfPictureContents))

            # open the relevant picture here...
            currentPicPath = pics_list[i]

            # get the original dimensions for conversion
            with Image.open(currentPicPath) as img:
                currImgWidth, currImgHeight = img.size

            #print(currImgWidth, currImgHeight)

            for foundObject in listOfPictureContents:
                imgClass = ourDict[int(foundObject[0])]
                relXMid = float(foundObject[1])
                relYMid = float(foundObject[2])
                relWidth = float(foundObject[3])
                relHeight = float(foundObject[4])

                relXTop = relXMid + relWidth/2

                relYTop = relYMid + relHeight/2

                relXBottom = relXMid - relWidth/2

                relYBottom = relYMid - relHeight/2

                xTop = relXTop * currImgWidth
                yTop = relYTop * currImgHeight
                xBottom = relXBottom * currImgWidth
                yBottom = relYBottom * currImgHeight

                ######### adapt path to dev box ###############


                currentPicPath = currentPicPath.replace('/media/korbinian/Elements/Korbinian_Bauer_Bachelorthesis/Datasets_v2/', '/home/robotics/Desktop/Korbinian_Bauer_Bachelorthesis/Datasets_v2/')


                ######################################

                stringForThisLine = currentPicPath+","+str(int(xBottom))+","+str(int(yBottom))+","+str(int(xTop))+","+str(int(yTop))+","+imgClass
                overallWriteString += stringForThisLine+"\n"

    destFileLabels = open(labelFileDest,'w')
    destFileLabels.write(overallWriteString)

    associationsString = ""
    for key in ourDict:
        associationsString+= ourDict[key]+","+str(key)+"\n"

    destFileAssociations = open(associationFileDest, 'w')
    destFileAssociations.write(associationsString)

print("Converting train labels to csv format...")
yolo2csv(directories.label_dir_train, directories.image_dir_train, 2, directories.annotations_file_train, directories.associations_file)
print("Converting validation labels to csv format...")
yolo2csv(directories.label_dir_val, directories.image_dir_val, 2, directories.annotations_file_val, directories.associations_file)
print("Generating Association file...")