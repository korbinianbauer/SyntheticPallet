import os
import glob
import directories

dirs = [directories.prediction_image_dir,
        directories.prediction_label_dir]

file_list = []

files_deleted = 0

for directory in dirs:
    files = glob.glob(directory+'*')
    for f in files:
        try:
            os.remove(f)
            files_deleted += 1
        except:
            print('Couldnt delete {}'.format(f))
            pass

for f in file_list:
    try:
        os.remove(f)
        files_deleted += 1
    except:
        print('Couldnt delete {}'.format(f))
        pass

print("Deleted {} files".format(files_deleted))