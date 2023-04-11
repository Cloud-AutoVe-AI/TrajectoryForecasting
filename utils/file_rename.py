from utils.functions import *
from shutil import copyfile

base_directory = 'D:/TrajectoryForecasting/DATASET/voss'
# folder_list = os.listdir(base_directory)
folder_list = ['0044', '0045', '0046', '0047', '0048', '0050', '0051', '0052']

for idx, folder_name in enumerate(folder_list):
    if (len(folder_name) > 4):
        continue

    target_directory = os.path.join(base_directory, folder_name)
    try:
        copyfile(os.path.join(target_directory, 'step2/label.csv'), os.path.join(target_directory, 'label_ori.csv'))
    except:
        copyfile(os.path.join(target_directory, 'step12/label.csv'), os.path.join(target_directory, 'label_ori.csv'))

    try:
        copyfile(os.path.join(target_directory, 'step3/label.csv'), os.path.join(target_directory, 'label_proc.csv'))
    except:
        print('>> step3 is not found ..')

    print(">> %s is done .." % folder_name)