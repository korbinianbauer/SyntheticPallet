import os
import glob
import directories

dirs = [directories.image_dir_train,
        directories.label_dir_train,
        directories.image_dir_val,
        directories.label_dir_val]

file_list = [directories.annotations_file_val,
         directories.annotations_file_train,
         directories.associations_file]

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

for directory in dirs:
    try:
        os.rmdir(directory)
        files_deleted += 1
    except:
        print('Couldnt delete {}'.format(directory))
        pass

for f in file_list:
    try:
        os.remove(f)
        files_deleted += 1
    except:
        print('Couldnt delete {}'.format(f))
        pass

print("Deleted {} files and directories".format(files_deleted))