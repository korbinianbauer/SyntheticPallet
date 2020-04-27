# To make system pythons packages accessible:
# With your os’ python, type, import cv2; cv2; and retrieve the path of where the module is installed.
# Then in blender, import sys; sys.path.append(the_path_of_the_module). And then you can import cv2


import sys
#sys.path.append('C:/Users/Besitzer/AppData/Roaming/Python/Python37/site-packages/cv2/') # make system python's openCV accessible
#sys.path.append('C:/Program Files/Python37/lib/site-packages/pascal_voc_writer/') # make system python's pascal-voc-writer accessible

import bpy
import bpy_extras
import mathutils
from random import randint, random, uniform, choice, seed
import time
import numpy as np
import os
import cv2
import pascal_voc_writer as voc
import glob



image_start_index = 0 # first image index to use, default 0
images = 100 # number of training samples to generate
hidden_BB_threshold = 0.1 # bounding boxes that are hidden by more than this fraction are not saved/drawn, comparable to retinanets non-maximum-suppression [0.0 ... 1.0]
label_probability = 0.5 # probability of a KLT having a label [0.0 ... 1.0]

# VARIATION SETTINGS

random_camera_position = False
camera_shift = [[-0.1, 0.1], [-1, -0.5], [-0.1, 0.1]]  # the maximum values [x,y,z] in meters to randomly shift the camera
camera_rotation = [1, 1, 2] # the maximum absolute value in degrees to randomly rotate the camera
default_camera_position = [-0.743, 0.30754, 1.2046]
default_camera_orientation = [91.4, 0, 0]

shift_klts = True
klt_shift = [0.03, 0.03, 0.02] # the maximum absolute value in meters to randomly shift individual KLTs
pallet_shift = [0.13, 0.1, 0.04] # the maximum absolute value in meters to randomly shift the whole pallet

random_light = True
light_size_range = [0.1, 5.0] # the minimum and maximum diameter in meters of each light source
light_power_range = [10, 100] # the minimum and maximum power in watts of each light source
light_pos_shift = 1.0 # the maximum absolute value in meters to randomly shift the individual light source
light_default_size = 3.0 # default diameter of light sources in meters

random_background = True


bb_overlap_delta_z_thres = 0.05 # When to consider a bounding box to be in front of (and hide) another one

background_image_path = 'C:/Users/Besitzer/Desktop/Synthetic Pallet/empty_background.png'
background_image_directory = 'C:/Users/Besitzer/Desktop/Synthetic Pallet/backgrounds/'
#background_image_path = 'C:/Users/Besitzer/Desktop/Synthetic Pallet/1724x1724.jpg'
foreground_directory = 'C:/Users/Besitzer/Desktop/Synthetic Pallet/blender_rendered/foreground/'
blended_directory = 'C:/Users/Besitzer/Desktop/Synthetic Pallet/blender_rendered/blended/'
bb_drawn_directory = 'C:/Users/Besitzer/Desktop/Synthetic Pallet/blender_rendered/bb_drawn/'
label_directory = 'C:/Users/Besitzer/Desktop/Synthetic Pallet/blender_rendered/labels/'

# clear console
os.system("cls")

# initialize random number generator
seed()

# read in set of background textures
background_images_glob = glob.glob(background_image_directory+"*.jpg")
background_images = []
for path in background_images_glob:
    path = path.replace("\\\\", "/")
    path = path.replace("\\", "/")
    background_images.append(path)
    
# KLT model and label and their relative position to each other, relevant volume to calculate the bounding boxes

KLT_template = bpy.data.objects['F-KLT 6410']
label_template = bpy.data.objects['6410-Label']
Camera_object = bpy.data.objects['Camera'] # camera object used for rendering training images

klt_zero_pos = [-1.014, 1.6199, 0.83733] # position of bottom-most, left-most, front-most KLT on pallet
klt_size = [0.592, 0.394, 0.264] # outer dimensions of the KLT, not including feet
klt_bb_relevant_volume = [[-0.592/2, 0.592/2], [-0.394/2, -0.394/2 + 0.02], [0, 0.264]]
klt_label_offset = [0, -0.1889, 0.0581] # the position of the label, relative to the KLT
label_bb_offset = [[-0.2/2, 0, 0.074/2], [0.2/2, 0, -0.074/2], [0.2/2, 0, 0.074/2], [-0.2/2, 0, -0.074/2]]
klt_pallet_capacity = [2,2,3] # maximum number of KLTs that fit on the pallet in each direction

# light sources and their default power and position
back_light_object = bpy.data.objects['Back Light']
back_light_default_power = 3.3
back_light_default_pos = [-0.743, 3.35, 1.275]

ceiling_light_object = bpy.data.objects['Ceiling Light']
ceiling_light_default_power = 173.2
ceiling_light_default_pos = [-0.743, 1.85, 2.633]

floor_light_object = bpy.data.objects['Floor Light']
floor_light_default_power = 2.8
floor_light_default_pos = [-0.743, 1.8624, -0.22]

front_light_object = bpy.data.objects['Front Light']
front_light_default_power = 13
front_light_default_pos = [-0.743, 0.22, 1.275]

left_light_object = bpy.data.objects['Left Light']
left_light_default_power = 3
left_light_default_pos = [-2.52, 1.771, 1.275]

right_light_object = bpy.data.objects['Right Light']
right_light_default_power = 24.1
right_light_default_pos = [1.0, 1.771, 1.275]

light_objects = [back_light_object, ceiling_light_object, floor_light_object, front_light_object, left_light_object, right_light_object]


def reset_camera_position():
    Camera_object.location = tuple(default_camera_position)
    
    # convert from degree to radians
    cam_orientation_rad = [val/360*2*3.141592 for val in default_camera_orientation]
    Camera_object.rotation_euler = tuple(cam_orientation_rad)
    
def randomize_camera_position():
    rand_pos = [uniform(default+offset[0], default+offset[1]) for default, offset in zip(default_camera_position, camera_shift)]
    rand_orientation = [uniform(default-offset, default+offset) for default, offset in zip(default_camera_orientation, camera_rotation)]
    rand_orientation_rad = [val/360*2*3.141592 for val in rand_orientation]
    
    Camera_object.location = tuple(rand_pos)
    Camera_object.rotation_euler = tuple(rand_orientation_rad)

def get_background_image():
    if random_background:
        image_path = choice(background_images)
        image = cv2.imread(image_path)
        
        #random crop
        height, width, depth = image.shape
        min_source_size = 50
        x_start = randint(0, width - min_source_size)
        x_end = randint(x_start + min_source_size, width)
        y_start = randint(0, height - min_source_size)
        y_end = randint(y_start + min_source_size, height)
        
        image = image[y_start:y_end, x_start:x_end]
        
        #resize to find render
        image = cv2.resize(image,(scn.render.resolution_x,scn.render.resolution_y))
        
        return image
        
    else:
        return cv2.imread(background_image_path)

def randomize_lights():
    for light_obj in light_objects:
        light_obj.data.energy = randint(*light_power_range)**3/light_power_range[1]**2
        light_obj.data.size = uniform(*light_size_range)**2/light_size_range[1]
        
    random_pos = [back_light_default_pos[i] + randint(-100*light_pos_shift, 100*light_pos_shift)/100.0 for i in range(3)]
    back_light_object.location = tuple(random_pos)
    
    random_pos = [ceiling_light_default_pos[i] + randint(-100*light_pos_shift, 100*light_pos_shift)/100.0 for i in range(3)]
    ceiling_light_object.location = tuple(random_pos)
    
    random_pos = [floor_light_default_pos[i] + randint(-100*light_pos_shift, 100*light_pos_shift)/100.0 for i in range(3)]
    floor_light_object.location = tuple(random_pos)
    
    random_pos = [front_light_default_pos[i] + randint(-100*light_pos_shift, 100*light_pos_shift)/100.0 for i in range(3)]
    front_light_object.location = tuple(random_pos)
    
    random_pos = [left_light_default_pos[i] + randint(-100*light_pos_shift, 100*light_pos_shift)/100.0 for i in range(3)]
    left_light_object.location = tuple(random_pos)
    
    random_pos = [right_light_default_pos[i] + randint(-100*light_pos_shift, 100*light_pos_shift)/100.0 for i in range(3)]
    right_light_object.location = tuple(random_pos)
        
def reset_lights():
    for light_obj in light_objects:
        light_obj.data.size = light_default_size
        
    back_light_object.location = tuple(back_light_default_pos)
    back_light_object.data.energy = back_light_default_power
    
    ceiling_light_object.location = tuple(ceiling_light_default_pos)
    ceiling_light_object.data.energy = ceiling_light_default_power
    
    floor_light_object.location = tuple(floor_light_default_pos)
    floor_light_object.data.energy = floor_light_default_power
    
    front_light_object.location = tuple(front_light_default_pos)
    front_light_object.data.energy = front_light_default_power
    
    left_light_object.location = tuple(left_light_default_pos)
    left_light_object.data.energy = left_light_default_power
    
    right_light_object.location = tuple(right_light_default_pos)
    right_light_object.data.energy = right_light_default_power
        
def get_random_pallet_configuration(klt_count = None):
    if klt_count is None:
        klt_count = randint(1, np.prod(klt_pallet_capacity))
        
    pallet = []
    
    while len(pallet) < klt_count:
        new_pos = get_random_klt_pos()
        if not new_pos in pallet:
            pallet.append(new_pos)
            
    # shift klts if activated
    if shift_klts:
        random_pallet_shift = [randint(-100*dist, 100*dist)/100.0 for dist in pallet_shift]
        for klt_index in range(len(pallet)):
            random_shift = [randint(-100*dist, 100*dist)/100.0 for dist in klt_shift]
            
            x,y,z = pallet[klt_index]
            x = x + random_shift[0] + random_pallet_shift[0]
            y = y + random_shift[1] + random_pallet_shift[1]
            z = z + random_shift[2] + random_pallet_shift[2]
            pallet[klt_index] = x,y,z
    
    return pallet
    

def get_random_klt_pos():
    x, y, z = [randint(1, cap) for cap in klt_pallet_capacity]
    
    return [x, y, z]

def get_klt_pos(zero_pos, size, x, y, z):
    coords = [zero_pos[i] + ([x, y, z][i]-1)*size[i] for i in range(len(size))]
    return coords

def alpha_blend(background, foreground):
    # Read the alpha chanel from foreground
    alpha = foreground[:,:,3]
    
    # remove alpha channel from foreground
    foreground = foreground[:,:,:3]
     
    # Convert uint8 to float
    foreground = foreground.astype(float)/255
    background = background.astype(float)/255
    
     
    # Normalize the alpha mask to keep intensity between 0 and 1
    alpha = alpha.astype(float)/255    
    
    for channel in range(3):
        # Multiply the foreground with the alpha matte
        foreground[:,:,channel] = cv2.multiply(alpha, foreground[:,:,channel])
        # Multiply the background with ( 1 - alpha )
        background[:,:,channel] = cv2.multiply(1.0 - alpha, background[:,:,channel])
        
    # Add the masked foreground and background.
    outImage = cv2.add(foreground, background)
    
    return outImage

def bb_intersection_over_area(boxA, boxB):
    #print("boxA: {}".format(boxA))
    #print("boxB: {}".format(boxB))
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[3], boxB[3])
    yB = min(boxA[4], boxB[4])
    
    #print("xA: {}, yA: {}, xB: {}, yB: {}".format(xA, yA, xB, yB))
 
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    #print("interArea: {}".format(interArea))
 
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[3] - boxA[0] + 1) * (boxA[4] - boxA[1] + 1))
    boxBArea = abs((boxB[3] - boxB[0] + 1) * (boxB[4] - boxB[1] + 1))
    #print("boxAArea: {} \nboxBArea: {}".format(boxAArea, boxBArea))
 
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea)
 
    # return the intersection over union value
    return iou

def filter_hidden_BBs(BBs, threshold):
    filtered = []
    
    for bb1 in range(len(BBs)):
        hidden = False
        for bb2 in range(len(BBs)):
            if bb1 == bb2: continue # don't compare BB with itself
            #print("Comparing BB {} with BB {}:".format(bb1, bb2))
            iou = bb_intersection_over_area(BBs[bb1], BBs[bb2])
            #print("IoU: {}".format(iou))
            if iou > threshold:
                # if bb1 is behind bb2, it's hidden
                if BBs[bb1][2] > BBs[bb2][2] + bb_overlap_delta_z_thres:
                    #print("BB{} is hidden by BB{}".format(bb1, bb2))
                    hidden = True
                
        if not hidden:
            #print("Appending")
            filtered.append(BBs[bb1])
        else:
            #print("Ignoring")
            pass
        
    return filtered

def get_bb_world_coords(object_coords, bb_offset):
    bb_p1 = [object_coords[0] + bb_offset[0][0], object_coords[1] + bb_offset[0][1], object_coords[2] + bb_offset[0][2]]
    bb_p2 = [object_coords[0] + bb_offset[1][0], object_coords[1] + bb_offset[1][1], object_coords[2] + bb_offset[1][2]]
    bb_p3 = [object_coords[0] + bb_offset[2][0], object_coords[1] + bb_offset[2][1], object_coords[2] + bb_offset[2][2]]
    bb_p4 = [object_coords[0] + bb_offset[3][0], object_coords[1] + bb_offset[3][1], object_coords[2] + bb_offset[3][2]]
    bb_world_coords = [bb_p1, bb_p2, bb_p3, bb_p4]
    return bb_world_coords

def get_bb_2D_coords(scene, camera_obj, world_coords):
    bb_2d_p1 = bpy_extras.object_utils.world_to_camera_view(scene, camera_obj, mathutils.Vector(world_coords[0]))
    bb_2d_p2 = bpy_extras.object_utils.world_to_camera_view(scene, camera_obj, mathutils.Vector(world_coords[1]))
    bb_2d_p3 = bpy_extras.object_utils.world_to_camera_view(scene, camera_obj, mathutils.Vector(world_coords[2]))
    bb_2d_p4 = bpy_extras.object_utils.world_to_camera_view(scene, camera_obj, mathutils.Vector(world_coords[3]))
    
    x_min = min(bb_2d_p1[0], bb_2d_p2[0], bb_2d_p3[0], bb_2d_p4[0])
    x_max = max(bb_2d_p1[0], bb_2d_p2[0], bb_2d_p3[0], bb_2d_p4[0])
    y_min = min(bb_2d_p1[1], bb_2d_p2[1], bb_2d_p3[1], bb_2d_p4[1])
    y_max = max(bb_2d_p1[1], bb_2d_p2[1], bb_2d_p3[1], bb_2d_p4[1])
    
    z = (bb_2d_p1[2] + bb_2d_p2[2] + bb_2d_p3[2] + bb_2d_p4[2]) / 4
    
    coords = [x_min, y_min, x_max, y_max, z]
    
    return coords

def get_bb_pixel_coords(res_x, res_y, render_scale, x_min, y_min, x_max, y_max, z):
    
    p1_pos_x = int(res_x * render_scale * x_min)
    p1_pos_y = res_y - int(res_y * render_scale * y_max)
    p1_pos_z = round(z, 3)
    p2_pos_x = int(res_x * render_scale * x_max)
    p2_pos_y = res_y - int(res_y * render_scale * y_min)
    p2_pos_z = round(z, 3)
    
    return p1_pos_x, p1_pos_y, p1_pos_z, p2_pos_x, p2_pos_y, p2_pos_z

def write_yolo_labelfile(path, BBs, x_dim, y_dim):
    label_file = open(path,'w')
    
    for BB in BBs:
        x_mid = (BB[0] + BB[3]) / 2 / x_dim
        y_mid = (BB[1] + BB[4]) / 2 / y_dim
        
        x_rel = abs(BB[0] - BB[3]) / x_dim
        y_rel = abs(BB[1] - BB[4]) / y_dim
        
        class_name = BB[7]
        
        label_file.write(str(class_name) + " " + str(round(x_mid,6)) + " " + str(round(y_mid,6)) + " " + str(round(x_rel,6)) + " " + str(round(y_rel,6)) + "\n")
    
    label_file.close()
    
    
def get_obj_bb(obj, scene, camera_obj):
    # get world coordinates of all vertices of object
    verts = [obj.matrix_world @ vert.co for vert in obj.data.vertices]
    
    # calculate the volumetric area that contains the relevant vertices for calculating the bounding box
    obj_origin = tuple(obj.location)
    volume_offset = klt_bb_relevant_volume
    relevant_volume_x = [obj_origin[0] + volume_offset[0][0], obj_origin[0] + volume_offset[0][1]]
    relevant_volume_y = [obj_origin[1] + volume_offset[1][0], obj_origin[1] + volume_offset[1][1]]
    relevant_volume_z = [obj_origin[2] + volume_offset[2][0], obj_origin[2] + volume_offset[2][1]]
    
    #print("KLT origin: {}".format(obj_origin))
    #print("relevant_volume_x: {}".format(relevant_volume_x))
    #print("relevant_volume_y: {}".format(relevant_volume_y))
    #print("relevant_volume_z: {}".format(relevant_volume_z))
    
    # filter all vertices that are not in the releveant volume
    filtered_verts = []
    filtered_verts_x = 0
    filtered_verts_y = 0
    filtered_verts_z = 0
    
    for coord in verts:
        if not (coord[0] >= relevant_volume_x[0] and coord[0] <= relevant_volume_x[1]):
            filtered_verts_x += 1
            continue
        if not (coord[1] >= relevant_volume_y[0] and coord[1] <= relevant_volume_y[1]):
            filtered_verts_y += 1
            continue
        if not (coord[2] >= relevant_volume_z[0] and coord[2] <= relevant_volume_z[1]):
            filtered_verts_z += 1
            continue
        
        
        filtered_verts.append(coord)
        
    #print("Total verts: {}".format(len(verts)))
    #print("Verts not fitting in x_limits: {}".format(filtered_verts_x))
    #print("Verts not fitting in y_limits: {}".format(filtered_verts_y))
    #print("Verts not fitting in z_limits: {}".format(filtered_verts_z))
    #print("Verts passed volumetric filter: {}".format(len(filtered_verts)))
        
    # calculate the image plane coordinates for all relevant vertices
    coords_2d = [bpy_extras.object_utils.world_to_camera_view(scene, camera_obj, coord) for coord in filtered_verts]
    
    x_vals = [coord[0] for coord in coords_2d]
    y_vals = [coord[1] for coord in coords_2d]
    z_vals = [coord[2] for coord in coords_2d]
    
    x_min = min(x_vals)
    x_max = max(x_vals)
    y_min = min(y_vals)
    y_max = max(y_vals)
    z = max(z_vals)
    
    
    coords = [x_min, y_min, x_max, y_max, z]
    
    return coords

# set up shortcuts to blender variables
scn = bpy.context.scene

start_time = time.time()

for image_count in range(image_start_index, images):
    
    image_start_time = time.time()
    
    # set image save path
    foreground_file_path = foreground_directory + 'render' + str(image_count).zfill(6) + '.png'
    blended_file_path = blended_directory + 'blended' + str(image_count).zfill(6) + '.png'
    bb_drawn_path = bb_drawn_directory + 'drawn' + str(image_count).zfill(6) + '.png'
    bpy.data.scenes['Scene'].render.filepath = foreground_file_path

    # generate a random pallet configuration
    random_pallet = get_random_pallet_configuration()
    
    # set camera position
    if random_camera_position:
        randomize_camera_position()
    else:
        reset_camera_position()

    # list of KLT objects on pallet
    KLTs = []
    
    # list of klt bounding boxes ("Ground truth")
    KLT_BBs = []
    
    # list of label objects on pallet
    labels = []
    
    # list of label bounding boxes ("Ground truth")
    label_BBs = []

    box_number = 0
    for pallet_pos in random_pallet:
        
        ############## KLTs ###############
        
        # create new copy of KLT template
        new_obj = KLT_template.copy()
        KLTs.append(new_obj)
        new_obj.data = KLT_template.data.copy()
        new_obj.animation_data_clear()
        bpy.context.collection.objects.link(new_obj)
        
        
        # move new KLT according to its position on the pallet
        new_klt_coords = tuple(get_klt_pos(klt_zero_pos, klt_size, *pallet_pos))
        new_obj.location = new_klt_coords
        
        # update the transformation matrix to make sure to get the correct vertex coordinates
        bpy.context.view_layer.update()

        #print('Added a new FKLT 6410 at pallet pos: ' +str(pallet_pos) + ('coords: ') + str(new_klt_coords))

        x_min, y_min, x_max, y_max, z = get_obj_bb(new_obj, scn, Camera_object)
        #print("x_min: {}, y_min: {}, x_max: {}, y_max: {}, z: {}".format(x_min, y_min, x_max, y_max, z))
        
        # Convert to pixel coordinates
        render_scale = scn.render.resolution_percentage / 100
        res_x = scn.render.resolution_x
        res_y = scn.render.resolution_y
        
        p1_pos_x, p1_pos_y, p1_pos_z, p2_pos_x, p2_pos_y, p2_pos_z = get_bb_pixel_coords(res_x, res_y, render_scale, x_min, y_min, x_max, y_max, z)
        
        #print(p1_pos_x, p1_pos_y, p1_pos_z, p2_pos_x, p2_pos_y, p2_pos_z)
        # Add bounding box to list
        KLT_BBs.append((p1_pos_x, p1_pos_y, p1_pos_z, p2_pos_x, p2_pos_y, p2_pos_z, box_number, "0"))
        
        
        ########### LABELS ##################
        
        if label_probability > random():
            
            # create new label object
            new_label = label_template.copy()
            labels.append(new_label)
            new_label.data = label_template.data.copy()
            new_label.animation_data_clear()
            bpy.context.collection.objects.link(new_label)
            
            # move new label according to its position on the pallet
            new_label_coords = tuple(np.array(new_klt_coords) + np.array(klt_label_offset))
            new_label.location = new_label_coords
            
            # update the transformation matrix to make sure to get the correct vertex coordinates
            bpy.context.view_layer.update()
            
            # calculate label Bounding Box world coordinates
            bb_world_coords = get_bb_world_coords(new_label_coords, label_bb_offset)

            # calculate label position in frame-relative 2D coords
            x_min, y_min, x_max, y_max, z = get_bb_2D_coords(scn, Camera_object, bb_world_coords)
            
            # Convert to pixel coordinates
            render_scale = scn.render.resolution_percentage / 100
            res_x = scn.render.resolution_x
            res_y = scn.render.resolution_y
            
            p1_pos_x, p1_pos_y, p1_pos_z, p2_pos_x, p2_pos_y, p2_pos_z = get_bb_pixel_coords(res_x, res_y, render_scale, x_min, y_min, x_max, y_max, z)
            
            # Add bounding box to list
            KLT_BBs.append((p1_pos_x, p1_pos_y, p1_pos_z, p2_pos_x, p2_pos_y, p2_pos_z, box_number, "4"))
            
        
        box_number += 1
        
    # set light sources
    if random_light:
        randomize_lights()
    else:
        reset_lights()
        
    

    # render and save the image
    bpy.context.scene.camera = Camera_object
    bpy.ops.render.render(write_still=True)
    
    # delete KLTs and labels to reset the scene
    for KLT in KLTs:
        bpy.data.objects.remove(KLT, do_unlink=True)
        
    for label in labels:
        bpy.data.objects.remove(label, do_unlink=True)
    
    # load background image
    background = get_background_image()

    
    # load foreground image (just rendered)
    foreground = cv2.imread(foreground_file_path, cv2.IMREAD_UNCHANGED)
    
    # blend background and foreground image
    blended = alpha_blend(background, foreground)
    
    # save blended image
    cv2.imwrite(blended_file_path, blended*255)
    
    # remove bounding boxes that are hidden by other bounding boxes
    KLT_BBs = filter_hidden_BBs(KLT_BBs, hidden_BB_threshold)
    
    # write label file    
    label_file_path = label_directory + "blended" + str(image_count).zfill(6) + '.txt'
    write_yolo_labelfile(label_file_path, KLT_BBs, scn.render.resolution_x, scn.render.resolution_y)
    
    
    # draw KLT bounding boxes on blended
    for x1, y1, z1, x2, y2, z2, box_number, object_class in KLT_BBs:
        BBs_drawn_hidden = cv2.rectangle(blended, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
        # Using cv2.putText() method 
        BBs_drawn_hidden = cv2.putText(BBs_drawn_hidden, "{}-{}".format(object_class, box_number), (int(x1+2), int(y1+12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
    
    # save blended image with bounding boxes
    cv2.imwrite(bb_drawn_path, BBs_drawn_hidden*255)

    image_end_time = time.time()
    
    images_done = image_count - image_start_index + 1
    time_gone = image_end_time - start_time
    time_per_image = time_gone/images_done
    images_left = images - image_count
    time_left = images_left * time_per_image

    print("Image Duration: ")
    print(str(image_end_time-image_start_time))
    print("\n")
    print("ETA: {}min\n\n".format(round(time_left/60, 2)))
    
        
    
    
end_time = time.time()

print("Duration: ")
print(str(end_time-start_time))
