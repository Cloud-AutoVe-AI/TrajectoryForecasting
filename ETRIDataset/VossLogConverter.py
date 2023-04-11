import numpy as np
import csv
from utils.libraries import *
from ETRIDataset.VossHelper import VossHelper, Pose


class DatasetConverter:

    def __init__(self, args):

        '''
        ------------------------------------------------------------------------------------------------
        frame index| class | obj_id | x[m] | y[m] |  z[m]  | heading[rad] | width[m] | length[m] | height[m] | lidar | image
        ------------------------------------------------------------------------------------------------
        0          | 1     | 2      | 3    | 4    |   5    |      6       |    7     | 8         | 9         | 10     | 11
        ------------------------------------------------------------------------------------------------
        '''

        self.dataset_path = args.dataset_path

    def read_raw_file(self, file_dir):
        return np.genfromtxt(file_dir, delimiter=',')

    def convert(self, target_scenes):

        for _, scene_name in enumerate(target_scenes):

            folder_path = os.path.join(self.dataset_path, scene_name)
            track_data = self.read_raw_file(os.path.join(folder_path, 'TRACK-RES.log'))[1:]
            lidar_timestamps = self.read_raw_file(os.path.join(folder_path, 'PANDAR64.csv'))[1:]
            image_timestamps = self.read_raw_file(os.path.join(folder_path, 'cameras00.csv'))[1:]
            gps_timestamps = self.read_raw_file(os.path.join(folder_path, 'GPS_FOG.csv'))[1:]

            timestamps = np.unique(track_data[:, 0])
            raw_data = np.concatenate([timestamps.reshape(timestamps.size, 1), np.zeros(shape=(timestamps.size, 3))], axis=1)
            for _, timestamp in enumerate(timestamps):

                minidx = np.argmin(np.abs(gps_timestamps[:, 0] - timestamp))
                z = gps_timestamps[minidx, 5]

                minidx = np.argmin(np.abs(lidar_timestamps[:, 0] - timestamp))
                lidar_file_name = lidar_timestamps[minidx, 1]

                minidx = np.argmin(np.abs(image_timestamps[:, 0] - timestamp))
                image_file_name = image_timestamps[minidx, 1]

                raw_data[_, 1] = z
                raw_data[_, 2] = lidar_file_name
                raw_data[_, 3] = image_file_name

            track_data_ext = np.zeros(shape=(track_data.shape[0], 12))
            track_data_ext[:, :5] = track_data[:, :5]
            track_data_ext[:, 6:10] = track_data[:, 5:]

            for i in range(track_data_ext.shape[0]):
                cur_timestamp = track_data_ext[i][0]
                corr_raw_data = raw_data[raw_data[:, 0] == cur_timestamp]
                track_data_ext[i][5] = corr_raw_data[0, 1]
                track_data_ext[i][-2] = corr_raw_data[0, 2]
                track_data_ext[i][-1] = corr_raw_data[0, 3]

            for index, timestamp in enumerate(timestamps):
                check = (track_data_ext == timestamp)
                track_data_ext[check] = index

            # save as csv file
            file_name = os.path.join(folder_path, 'label.csv')
            fp = open(file_name, 'w')
            csvWriter = csv.writer(fp, lineterminator='\n')
            csvWriter.writerow(['frame index', 'class', 'obj_id', 'x[m]', 'y[m]', 'z[m]', 'heading[rad]', 'width[m]', 'length[m]', 'height[m]', 'lidar file name', 'image file name'])

            for i in range(track_data_ext.shape[0]):
                cur_data = track_data_ext[i]
                cur_line = [str(int(cur_data[0]))] # frame index
                cur_line.append(str(int(cur_data[1]))) # class
                cur_line.append(str(int(cur_data[2])))  # obj id
                cur_line.append(str(np.around(cur_data[3], decimals=5)))  # x
                cur_line.append(str(np.around(cur_data[4], decimals=5)))  # y
                cur_line.append(str(np.around(cur_data[5], decimals=5)))  # z
                cur_line.append(str(np.around(cur_data[6], decimals=5)))  # heading
                cur_line.append(str(np.around(cur_data[7], decimals=5)))  # w
                cur_line.append(str(np.around(cur_data[8], decimals=5)))  # l
                cur_line.append(str(np.around(cur_data[9], decimals=5)))  # h
                lidar_file_name = '%08d' % cur_data[10]
                image_file_name = '%08d' % cur_data[11]
                cur_line.append(lidar_file_name)  # lidar
                cur_line.append(image_file_name)  # image
                csvWriter.writerow(cur_line)
            fp.close()

            from shutil import copyfile
            source = '/home/dooseop/DATASET/voss/cam0_extcalib.csv'
            destination = os.path.join(folder_path, 'cam0_extcalib.csv')
            copyfile(source, destination)

            print(">> %s is done .." % scene_name)



def main():

    abspath = os.path.dirname(os.path.realpath(__file__))
    os.chdir(Path(abspath).parent.absolute())

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', type=str, default='/home/dooseop/DATASET/voss')
    args = parser.parse_args()

    DC = DatasetConverter(args)
    DC.convert(['0106', '0107', '0108', '0109', '0110', '0111', '0119', '0123', '0124', '0125', '0126', '0127', '0128', '0132',
                '0135', '0136', '0137', '0138', '0139', '0140', '0141', '0142', '0143', '0144', '0145', '0146'])

if __name__ == '__main__':
    main()
