import numpy as np
import h5py

##############################read files from downloading#######################################
f = h5py.File('GOLD_XYZ_OSC.0001_1024.hdf5', 'r')
dir_path = 'ExtractDataset'
modu_snr_size = 1200
############################################################################
"""
sampling the dataset
"""
for modu in range(24):
    X_list = []
    Y_list = []
    Z_list = []
    print('part ', modu)
    start_modu = modu * 106496
    for snr in range(26):
        start_snr = start_modu + snr * 4096
        idx_list = np.random.choice(range(0, 4096), size=modu_snr_size, replace=False)
        X = f['X'][start_snr:start_snr + 4096][idx_list]
        X_list.append(X)
        Y_list.append(f['Y'][start_snr:start_snr + 4096][idx_list])
        Z_list.append(f['Z'][start_snr:start_snr + 4096][idx_list])

    filename = dir_path + '/part' + str(modu) + '.h5'
    fw = h5py.File(filename, 'w')
    fw['X'] = np.vstack(X_list)
    fw['Y'] = np.vstack(Y_list)
    fw['Z'] = np.vstack(Z_list)
    print('X shape:', fw['X'].shape)
    print('Y shape:', fw['Y'].shape)
    print('Z shape:', fw['Z'].shape)
    fw.close()
f.close()
