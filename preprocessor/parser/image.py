import os
import pickle
import gzip
import numpy as np
from scipy import ndimage, misc
from sklearn import decomposition, preprocessing
from .util import numerical_sort
from preprocessor import settings

def get_image_paths(path):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(os.path.abspath(path) + '/raygrid'):
        files.extend(filenames)
        break

    files = sorted(files, key=numerical_sort)
    files = [path + '/raygrid/' + f for f in files]
    return files


def crop(path, frame_data):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(os.path.abspath(path) + '/raygrid'):
        files.extend(filenames)
        break

    files = sorted(files, key=numerical_sort)
    files = [path + '/raygrid/' + f for f in files]

    if not os.path.exists(path + '/raygrid/cropped/'):
        os.makedirs(path + '/raygrid/cropped/')

    for idx, f in enumerate(files):
        print('cropping %s' % f)
        print('   to %s' % os.path.dirname(f) + '/cropped/' + os.path.splitext(os.path.basename(f))[0] + '.png')

        img = ndimage.imread(f, flatten=True)
        xc = int(frame_data[idx, 7])
        yc = int(frame_data[idx, 8])
        h, w = img.shape

        cropped = np.empty((128, 128), dtype=int)
        for yi, y in enumerate(range(yc-64, yc+64)):
            for xi, x in enumerate(range(xc-64, xc+64)):
                nx, ny = x, y
                if nx < 0:
                    nx = nx + w
                if nx > w:
                    nx = nx - w
                if ny < 0:
                    ny = ny + h
                if ny > h:
                    ny = ny - h

                cropped[yi, xi] = img[ny, nx]

        misc.imsave(os.path.dirname(f) + '/cropped/' + os.path.splitext(os.path.basename(f))[0] + '.png', cropped)


def parse_images(path, base_name, frame_data):
    f_x = gzip.open('data/' + base_name + '.diff.x.dat', 'wb+')
    f_y = gzip.open('data/' + base_name + '.diff.y.dat', 'wb+')

    files = []
    for (dirpath, dirnames, filenames) in os.walk(os.path.abspath(path) + '/raygrid'):
        files.extend(filenames)
        break

    files = sorted(files, key=numerical_sort)

    first_frame = ndimage.imread(path + '/raygrid/' + files[0], flatten=True)
    pca = decomposition.PCA(n_components=settings.n_pca_components, copy=False)

    if settings.use_pca:
        first_frame_pca = pca.fit_transform(first_frame)
        dim = first_frame_pca.shape[0] * first_frame_pca.shape[1]
    else:
        dim = first_frame.shape[0] * first_frame.shape[1]

    x = np.zeros((1, dim + 4))
    y = np.zeros((1, dim))

    x[0, -4:] = frame_data[0, [3, 4, 5, 6]]
    pickle.dump(x, f_x, pickle.HIGHEST_PROTOCOL)
    pickle.dump(y, f_y, pickle.HIGHEST_PROTOCOL)

    last_y = first_frame
    last_y_diff = np.zeros_like(y)

    for idx in range(1, len(files)):
        print(f'Processing image {files[idx]}')

        y = ndimage.imread(path + '/raygrid/' + files[idx], flatten=True)
        diff = np.subtract(y, last_y).astype(float)
        diff = preprocessing.normalize(diff)

        if settings.save_diffs:
            iout = misc.toimage(diff)
            iout.save(path + '/diff/' + str(idx) + '.jpg')

        if settings.use_pca:
            diff = pca.fit_transform(diff)
            diff = diff.flatten()
            diff = diff.reshape((1, diff.shape[0]))
        else:
            diff = diff.flatten()
            diff = diff.reshape((1, diff.shape[0]))

        if settings.save_diffs:
            iout = misc.toimage(diff.reshape((600, 100)))
            iout.save(path + '/diff/' + str(idx) + '.pca.jpg')

        pickle.dump(diff, f_y, pickle.HIGHEST_PROTOCOL)
        last_y = y

        x[0, :-4] = last_y_diff
        x[0, -4:] = preprocessing.minmax_scale(frame_data[idx, [3, 4, 5, 6]], feature_range=(-1, 1))
        pickle.dump(x, f_x, pickle.HIGHEST_PROTOCOL)

        last_y_diff = diff

    f_y.close()
    f_x.close()


def load(filename):
    with gzip.open(filename, 'rb') as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


def image_generator(base_name, dim, batch_size):
    while True:
        for batch in range(0, 32000//batch_size):
            x = np.array([]).reshape(0, dim + 4)
            y = np.array([]).reshape(0, dim)
            x_in = load('data/' + base_name + '.diff.x.dat')
            y_in = load('data/' + base_name + '.diff.y.dat')

            for _ in range(batch * batch_size, (batch + 1) * batch_size):
                x = np.append(x, next(x_in), axis=0)
                n = next(y_in)
                n = n.reshape((1, 60000))
                y = np.append(y, n, axis=0)

            yield x, y
