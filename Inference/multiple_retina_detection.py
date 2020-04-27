from __future__ import print_function
import os
import sys
import singleRetinaDetection
import directories
import time

names_list = [os.path.splitext(file)[0] for file in os.listdir(directories.input_image_dir) if file.endswith('.png')]

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

predictions_done = 0
start_time = time.time()

for i in range(0, len(names_list)):
    image_name = directories.input_image_dir + names_list[i] + '.png'
    detections = singleRetinaDetection.detect_img(image_name, directories.prediction_image_dir, names_list[i])

    #print(detections)

    with open(directories.prediction_label_dir + "/prediction_" + names_list[i] + ".txt", "w") as f:
        
        for detection in detections:

            if detection['ObjectClassName'] == 'KLT_6410':
                class_index = '0'
            elif detection['ObjectClassName'] == 'label':
                class_index = '4'
            else:
                print("UNKOWN CLASS INDEX")

            f.write(class_index + " " + str(detection['x_min']) + " " + str(detection['x_max']) + " " + str(detection['y_min']) + " " + str(detection['y_max']) + " " + str(detection['confidence']) + "\n")
            

    f.close()

    predictions_done += 1
    predictions_left = len(names_list) - predictions_done
    time_per_prediction = (time.time() - start_time) / predictions_done
    time_left = int(predictions_left / time_per_prediction)
    time_left_string = time.strftime('%H:%M:%S', time.gmtime(time_left))

    print(bcolors.OKBLUE + 'Inference done for {}/{} samples. ETA: {}s'.format(predictions_done, len(names_list), time_left_string) + bcolors.ENDC, end = '\r')
    sys.stdout.flush()

print(bcolors.OKGREEN + bcolors.BOLD + "\nInference done." + bcolors.ENDC)
