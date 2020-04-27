import os

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

use_default_dirs = ('y' == raw_input("Do you want to use default direcotries? y/n\n"))

if not use_default_dirs:
    custom_base_dir = raw_input("Please enter the path to the folder containing 'raw'\n")

    custom_base_dir = custom_base_dir.strip()
    custom_base_dir = custom_base_dir.replace("file://", "")


    dirs_file = open('directories.py', "w")

    dirs = list()
    dirs.append("image_dir_input = '" + custom_base_dir + "/raw/images/'")
    dirs.append("label_dir_input = '" + custom_base_dir + "/raw/labels/'")
    dirs.append("image_dir_train = '" + custom_base_dir + "/images_train/'")
    dirs.append("label_dir_train = '" + custom_base_dir + "/labels_train/'")
    dirs.append("image_dir_val = '" + custom_base_dir + "/images_val/'")
    dirs.append("label_dir_val = '" + custom_base_dir + "/labels_val/'")
    dirs.append("annotations_file_val = '" + custom_base_dir + "/annotations_val.csv'")
    dirs.append("annotations_file_train = '" + custom_base_dir + "/annotations_train.csv'")
    dirs.append("associations_file = '" + custom_base_dir + "/associations.csv'")

    dirs_file.writelines('\n'.join(dirs) + '\n')
    dirs_file.close()


else:
    default_dirs_file = open('directories default.py', "r")
    dirs_file = open('directories.py', "w")
    default_dirs = default_dirs_file.readlines()

    dirs_file.writelines(default_dirs)

    default_dirs_file.close()
    dirs_file.close()


os.system("clear")
print(bcolors.OKGREEN + "Emptying output directory" + bcolors.ENDC)
os.system("python empty_output.py")
print(bcolors.OKGREEN + "Done. \n\nSplitting dataset" + bcolors.ENDC)
os.system("python split_dataset.py")
print(bcolors.OKGREEN + "Done. \n\nAugmenting dataset" + bcolors.ENDC)
os.system("python augmentation_multithreading.py")
print(bcolors.OKGREEN + "Done. \n\nConverting to csv format" + bcolors.ENDC)
os.system("python y2c.py")
print(bcolors.OKGREEN + "Done\n" + bcolors.BOLD + "Preparation done." + bcolors.ENDC)

if use_default_dirs:
    yes = {'yes_please'}
    no = {'no','n'}

    print("Do you want to clear the input directory? yes_please/no")
    choice = raw_input().lower()
    if choice in yes:
        os.system("python empty_input.py")
    elif choice in no:
        pass
    else:
        print("Please respond with 'y' or 'n'")