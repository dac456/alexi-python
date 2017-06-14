import os
import pickle
import gzip
import numpy as np
from scipy import ndimage, misc
from sklearn import decomposition, preprocessing
from .util import numerical_sort
from preprocessor import settings


class ImageProcessor:
    def __init__(self, base_name, path, frame_data, excluded_indices):
        self.base_name = base_name
        self.path = path
        self.frame_data = frame_data
        self.excluded = excluded_indices

    def crop(self):
        files = []
        for _, _, filenames in os.walk(os.path.abspath(self.path) + '/raygrid'):
            files.extend(filenames)
            break

        files = sorted(files, key=numerical_sort)
        files = [self.path + '/raygrid/' + f for f in files]
        print(f'len(files): {len(files)}')
        print(f'len(frame_data): {self.frame_data.shape[0]}')

        if not os.path.exists(self.path + '/raygrid/cropped/'):
            os.makedirs(self.path + '/raygrid/cropped/')

        for idx, f in enumerate(files):
            if idx not in self.excluded:
                # print('cropping %s' % f)
                # print('   to %s' % os.path.dirname(f) + '/cropped/' + os.path.splitext(os.path.basename(f))[0] + '.png')

                img = ndimage.imread(f, flatten=True)
                xc = int(self.frame_data[idx, 7])
                yc = int(self.frame_data[idx, 8])
                h, w = img.shape

                half_size = [x // 2 for x in settings.crop_dimensions]
                cropped = np.empty(settings.crop_dimensions, dtype=int)
                for yi, y in enumerate(range(yc - half_size[1], yc + half_size[1])):
                    for xi, x in enumerate(range(xc - half_size[0], xc + half_size[0])):
                        nx, ny = x, y
                        if nx < 0:
                            nx = nx + w
                        if nx > w - 1:
                            nx = nx - w
                        if ny < 0:
                            ny = ny + h
                        if ny > h - 1:
                            ny = ny - h

                        cropped[yi, xi] = img[ny, nx]

                misc.imsave(os.path.dirname(f) + '/cropped/' + os.path.splitext(os.path.basename(f))[0] + '.png', cropped)

    def parse(self):
        f_x = gzip.open('data/' + self.base_name + '.diff.x.dat', 'ab+')
        f_y = gzip.open('data/' + self.base_name + '.diff.y.dat', 'ab+')

        files = []
        for _, _, filenames in os.walk(os.path.abspath(self.path) + '/raygrid/cropped'):
            files.extend(filenames)
            break

        files = sorted(files, key=numerical_sort)

        first_frame = ndimage.imread(self.path + '/raygrid/cropped/' + files[0], flatten=True)
        pca = decomposition.PCA(n_components=settings.n_pca_components, copy=False)

        x = np.zeros_like(first_frame)
        y = np.zeros_like(first_frame)

        pickle.dump(x, f_x, pickle.HIGHEST_PROTOCOL)
        pickle.dump(y, f_y, pickle.HIGHEST_PROTOCOL)

        last_y = first_frame
        last_y_diff = np.zeros_like(y)

        for idx in range(1, len(files)):
            if idx not in self.excluded:
                print(f'Processing image {files[idx]}')

                y = ndimage.imread(self.path + '/raygrid/cropped/' + files[idx], flatten=True)
                diff = np.subtract(y, last_y).astype(float)
                # diff = preprocessing.normalize(diff)

                if settings.save_diffs:
                    iout = misc.toimage(diff)
                    iout.save(self.path + '/diff/' + str(idx) + '.jpg')

                if settings.use_pca:
                    diff = pca.fit_transform(diff)
                    # diff = diff.flatten()
                    # diff = diff.reshape((1, diff.shape[0]))
                # else:
                    # diff = diff.flatten()
                    # diff = diff.reshape((1, diff.shape[0]))

                if settings.save_diffs and settings.use_pca:
                    iout = misc.toimage(diff.reshape((600, 100)))
                    iout.save(self.path + '/diff/' + str(idx) + '.pca.jpg')

                pickle.dump(diff, f_y, pickle.HIGHEST_PROTOCOL)
                last_y = y

                x = last_y_diff
                pickle.dump(x, f_x, pickle.HIGHEST_PROTOCOL)

                last_y_diff = diff

        f_y.close()
        f_x.close()


def crop(paths, frame_data, slices):
    for path_idx, path in enumerate(paths):
        files = []
        for _, _, filenames in os.walk(os.path.abspath(path) + '/raygrid'):
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
            xc = int(frame_data[idx + slices[path_idx], 7])
            yc = int(frame_data[idx + slices[path_idx], 8])
            h, w = img.shape

            half_size = [x//2 for x in settings.crop_dimensions]
            cropped = np.empty(settings.crop_dimensions, dtype=int)
            for yi, y in enumerate(range(yc-half_size[1], yc+half_size[1])):
                for xi, x in enumerate(range(xc-half_size[0], xc+half_size[0])):
                    nx, ny = x, y
                    if nx < 0:
                        nx = nx + w
                    if nx > w - 1:
                        nx = nx - w
                    if ny < 0:
                        ny = ny + h
                    if ny > h - 1:
                        ny = ny - h

                    cropped[yi, xi] = img[ny, nx]

            misc.imsave(os.path.dirname(f) + '/cropped/' + os.path.splitext(os.path.basename(f))[0] + '.png', cropped)


def parse_images(paths, base_name, frame_data, slices):
    f_x = gzip.open('data/' + base_name + '.diff.x.dat', 'wb+')
    f_y = gzip.open('data/' + base_name + '.diff.y.dat', 'wb+')

    for path_idx, path in enumerate(paths):
        files = []
        for (dirpath, dirnames, filenames) in os.walk(os.path.abspath(path) + '/raygrid/cropped'):
            files.extend(filenames)
            break

        files = sorted(files, key=numerical_sort)

        first_frame = ndimage.imread(path + '/raygrid/cropped/' + files[0], flatten=True)
        pca = decomposition.PCA(n_components=settings.n_pca_components, copy=False)

        if settings.use_pca:
            first_frame_pca = pca.fit_transform(first_frame)
            dim = first_frame_pca.shape[0] * first_frame_pca.shape[1]
        else:
            dim = first_frame.shape[0] * first_frame.shape[1]

        x = np.zeros((1, dim + 4))
        y = np.zeros((1, dim))

        x[0, -4:] = frame_data[0 + slices[path_idx], [3, 4, 5, 6]]
        pickle.dump(x, f_x, pickle.HIGHEST_PROTOCOL)
        pickle.dump(y, f_y, pickle.HIGHEST_PROTOCOL)

        last_y = first_frame
        last_y_diff = np.zeros_like(y)

        for idx in range(1, len(files)):
            print(f'Processing image {files[idx]}')

            y = ndimage.imread(path + '/raygrid/cropped/' + files[idx], flatten=True)
            diff = np.subtract(y, last_y).astype(float)
            # diff = preprocessing.normalize(diff)

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

            if settings.save_diffs and settings.use_pca:
                iout = misc.toimage(diff.reshape((600, 100)))
                iout.save(path + '/diff/' + str(idx) + '.pca.jpg')

            pickle.dump(diff, f_y, pickle.HIGHEST_PROTOCOL)
            last_y = y

            x[0, :-4] = last_y_diff
            x[0, -4:] = frame_data[idx + slices[path_idx], [3, 4, 5, 6]]
            pickle.dump(x, f_x, pickle.HIGHEST_PROTOCOL)

            last_y_diff = diff

    f_y.close()
    f_x.close()


def load(filename, zipped=True):
    if zipped:
        with gzip.open(filename, 'rb') as f:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    break
    else:
        with open(filename, 'rb') as f:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    break


def image_generator(base_name, dim, batch_size, num_samples=186168):
    with open(f'data/{base_name}.framedata.scaler_in.dat', 'rb') as f:
        framedata_scaler_in = pickle.load(f)

    while True:
        for batch in range(0, num_samples//batch_size):
            x = np.array([]).reshape(0, dim + 4)
            y = np.array([]).reshape(0, dim)
            x_in = load(f'data/{base_name}.diff.x.dat')
            y_in = load(f'data/{base_name}.diff.y.dat')
            frame_in = load(f'data/{base_name}.framedata_img.dat', zipped=False)

            for _ in range(batch * batch_size, (batch + 1) * batch_size):
                imgx = np.reshape(next(x_in), (1, dim))
                imgx = preprocessing.minmax_scale(imgx, feature_range=(-1, 1), axis=None)
                framex = np.reshape(next(frame_in)[4:8], (1, 4))
                framex = framedata_scaler_in.transform(framex)
                xc = np.concatenate([imgx, framex], axis=1)
                x = np.append(x, xc, axis=0)
                n = next(y_in)
                n = n.reshape((1, dim))
                n = preprocessing.minmax_scale(n, feature_range=(-1, 1), axis=None)
                y = np.append(y, n, axis=0)

            yield x, y
