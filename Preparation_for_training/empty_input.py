import os
import glob
import directories

dirs = [directories.image_dir_input,
        directories.label_dir_input]

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

print("Deleted {} files".format(files_deleted))