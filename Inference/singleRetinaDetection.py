import keras
import tensorflow as tf
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image

import numpy as np
from datetime import datetime
import cv2

import directories


# set tf backend to allow memory to grow, instead of claiming everything
def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

# load Retinanet model
model = models.load_model(directories.weights_file, backbone_name='resnet50')

# Convert to an inference model, to get rid of the useless part of the model, optimizing its performance !
model = models.convert_model(model)

labels_to_names = {0: 'KLT_6410', 4: 'label', 1: 'KLT_6147', 2: 'KLT_6412', 3: 'KLT_4147'}


"""to make it thread-safe"""
model._make_predict_function()


def detect_img(current_img_path, save_img_path, img_name):
    img = cv2.imread(current_img_path, 1)
    image = read_image_bgr(current_img_path)
    image = preprocess_image(image)
    image, scale = resize_image(image)

    # RAW PREDICTION. Interestingly, the model is only really loaded after the first prediction. Do it with a trivial picture first perhaps...

    start = datetime.now()

    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
                                                                       # Axis of the different pictures, which are encoded as ... what?
    print('processing time = ' + str(datetime.now() - start))

    # Correct for image scale
    boxes /= scale
    detections = []
    boxes = np.array(boxes)
    labels = np.array(labels)

    confidence_thresh = 0.01

    # Determines which boxes are shown - index filtering
    good_scores = np.array(scores) > confidence_thresh

    shown_boxes = boxes[good_scores]
    shown_labels = labels[good_scores]
    shown_confidences = scores[good_scores]

    #print("labels:" + str(labels))
    #print("goodScores:" + str(good_scores))
    #print("shownLabels:" + str(shown_labels))

    """
#   NOT MODULAR CODE!!!
    although python stretches the array automatically if nonoverlapping
    """
    shown_label_names = [labels_to_names[la] for la in shown_labels]

    for i in range(len(shown_boxes)):

        curr_label_name = shown_label_names[i]
        x_min = shown_boxes[i][0]
        y_min = shown_boxes[i][1]
        x_max = shown_boxes[i][2]
        y_max = shown_boxes[i][3]
        confidence = shown_confidences[i]
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, curr_label_name + " " + str(confidence), (x_min, y_min), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        detections.append(
            {"ObjectClassName": curr_label_name,  "confidence": float(confidence), "x_min": int(round(x_min)), "y_min": int(round(y_min)), "x_max": int(round(x_max)), "y_max": int(round(y_max))})

    cv2.imwrite(save_img_path + img_name + '.png', img)

    return detections
