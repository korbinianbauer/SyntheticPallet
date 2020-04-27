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

os.system("clear")
print(bcolors.OKGREEN + "Emptying output directory" + bcolors.ENDC)
os.system("python empty_output.py")
print(bcolors.OKGREEN + "Done. \n\nSplitting dataset" + bcolors.ENDC)
os.system("python split_dataset.py")
print(bcolors.OKGREEN + "Done. \n\nAugmenting dataset" + bcolors.ENDC)
os.system("python augmentation.py")
print(bcolors.OKGREEN + "Done. \n\nConverting to csv format" + bcolors.ENDC)
os.system("python y2c.py")
print(bcolors.OKGREEN + "Done\n" + bcolors.BOLD + "Preparation done." + bcolors.ENDC)