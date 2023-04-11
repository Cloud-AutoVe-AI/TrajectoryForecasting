from utils.libraries import *

def save_as_bin_file_4d(map_data, file_path, dsize):

    bs, ch, ht, wt = map_data.shape

    binfile = open(file_path, 'wb')
    for b in range(bs):
        for c in range(ch):
            for h in range(ht):
                for w in range(wt):
                    data = struct.pack(dsize, map_data[b, c, h, w])  # pack 'num' with a integer size
                    binfile.write(data)

    binfile.close()

def save_as_bin_file_3d(map_data, file_path, dsize):

    bs, ch, ht = map_data.shape

    binfile = open(file_path, 'wb')
    for b in range(bs):
        for c in range(ch):
            for h in range(ht):
                 data = struct.pack(dsize, map_data[b, c, h])  # pack 'num' with a integer size
                 binfile.write(data)

    binfile.close()

def save_as_bin_file_2d(map_data, file_path, dsize):

    bs, ch = map_data.shape

    binfile = open(file_path, 'wb')
    for b in range(bs):
        for c in range(ch):
             data = struct.pack(dsize, map_data[b, c])  # pack 'num' with a integer size
             binfile.write(data)

    binfile.close()

def save_as_bin_file_1d(map_data, file_path, dsize):

    bs, ch = map_data.shape

    binfile = open(file_path, 'wb')
    for b in range(bs):
        for c in range(ch):
             data = struct.pack(dsize, map_data[b, c])  # pack 'num' with a integer size
             binfile.write(data)

    binfile.close()

def read_bin_file_4d(file_path, dsize, shape):

    bs, ch, ht, wt = shape
    output = np.zeros(shape=(bs, ch, ht, wt))

    binfile = open(file_path, 'rb')
    floatsize = struct.calcsize(dsize)

    for b in range(bs):
        for c in range(ch):
            for h in range(ht):
                for w in range(wt):
                    data = binfile.read(floatsize)
                    output[b, c, h, w] = struct.unpack(dsize, data)[0]

    return output

def read_bin_file_3d(file_path, dsize, shape):

    bs, ch, ht = shape
    output = np.zeros(shape=(bs, ch, ht))

    binfile = open(file_path, 'rb')
    floatsize = struct.calcsize(dsize)

    for b in range(bs):
        for c in range(ch):
            for h in range(ht):
                data = binfile.read(floatsize)
                output[b, c, h] = struct.unpack(dsize, data)[0]

    return output

def read_bin_file_2d(file_path, dsize, shape):

    bs, ch = shape
    output = np.zeros(shape=(bs, ch))

    binfile = open(file_path, 'rb')
    floatsize = struct.calcsize(dsize)

    for b in range(bs):
        for c in range(ch):
            data = binfile.read(floatsize)
            output[b, c] = struct.unpack(dsize, data)[0]

    return output
