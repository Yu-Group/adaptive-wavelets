import os

import numpy as np

opj = os.path.join
from astropy.io import fits

data_path = '../../data/cosmology'


def downsample(parameter_file, root_dir, resize=64, nsamples=30000, ncosmo=10):
    '''
    downsample cosmolgy image
    '''
    print('preprocessing...')
    img_size = 256
    params_ = np.loadtxt(parameter_file)[:ncosmo]

    image = []
    params = []
    for idx in range(nsamples):
        img_name = opj(root_dir, 'model%03d/WLconv_z1.00_%04dr.fits' % (idx % len(params_), idx // len(params_)))
        start1 = np.random.randint(0, img_size - resize - 1, 1)[0]
        start2 = np.random.randint(0, img_size - resize - 1, 1)[0]
        end1 = start1 + resize
        end2 = start2 + resize
        hdu_list = fits.open(img_name, memmap=False)
        img_data = hdu_list[0].data
        image.append(img_data[start1:end1, start2:end2])
        hdu_list.close()
        params.append(params_[idx % len(params_), 1:-1])
        print('\r idx: {}/{}'.format(idx, nsamples), end='')

    image = np.stack(image, axis=0)
    params = np.stack(params, axis=0)
    # save
    np.savez(opj(data_path, 'cosmo_resize_{}.npz'.format(resize)), imgs=image, params=params)


if __name__ == '__main__':
    parameter_file = opj(data_path, 'cosmological_parameters.txt')
    root_dir = opj(data_path, 'z1_256')
    resize = 64

    # save
    downsample(parameter_file, root_dir, resize)

    # load
    dataset_zip = np.load(opj(data_path, 'cosmo_resize_{}.npz'.format(resize)))
    imgs = dataset_zip['imgs']
    params = dataset_zip['params']
    print(imgs.shape, params.shape)
