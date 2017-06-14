import sys
import getopt
import pickle
import datetime as dt
import pygame as pg
import numpy as np
from keras.models import load_model
from preprocessor import settings

from sim.vehicle import Vehicle

sys.setrecursionlimit(40000)

path, name = '', ''
opts, args = getopt.getopt(sys.argv[1:], 'p:n:')
for o, a in opts:
    if o == '-p':
        path = a
    if o == '-n':
        name = a


pg.init()
screen = pg.display.set_mode(settings.dimensions)

models = {
    'terrain': load_model(f'networks/{name}.terrain.h5'),
    'dx': load_model(f'networks/{name}.dx.h5'),
    'dy': load_model(f'networks/{name}.dy.h5'),
    'v': load_model(f'networks/{name}.speed.h5'),
    'dtheta': load_model(f'networks/{name}.dtheta.h5')
}

terrain = pg.image.load(path + '/raygrid/frame0.tga')

with open(f'data/{name}.framedata.scaler_in.dat', 'rb') as f:
    framedata_scaler_in = pickle.load(f)

with open(f'data/{name}.framedata.scaler_out.dat', 'rb') as f:
    framedata_scaler_out = pickle.load(f)


img_dim = settings.crop_dimensions[0] * settings.crop_dimensions[1]
diff = np.zeros((600, 600))
last = None

vehicle = Vehicle(terrain, radius=64)

while True:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            sys.exit()

    start_frame = dt.datetime.now()
    t = np.mean(pg.surfarray.array3d(terrain), -1)
    if last is not None:
        diff = (t - last).astype(float)
        # diff = diff.flatten()
        # diff = diff.reshape((1, diff.shape[0]))

    half_size = [x//2 for x in settings.crop_dimensions]
    cropped = np.empty(settings.crop_dimensions, dtype=int)
    for yi, y in enumerate(range(int(vehicle.position[1]) - half_size[1], int(vehicle.position[1]) + half_size[1])):
        for xi, x in enumerate(range(int(vehicle.position[0]) - half_size[0], int(vehicle.position[0]) + half_size[0])):
            nx, ny = x, y
            if nx < 0:
                nx += 600
            if nx > 600 - 1:
                nx -= 600
            if ny < 0:
                ny += 600
            if ny > 600 - 1:
                ny -= 600

            cropped[yi, xi] = diff[ny, nx]

    cropped = cropped.flatten()
    cropped = cropped.reshape((1, cropped.shape[0]))

    diff_in = np.zeros((1, img_dim + 4))
    diff_in[0, :-4] = cropped
    vehicle.step(models, diff_in, framedata_scaler_in, framedata_scaler_out)

    last = t

    screen.blit(terrain, (0, 0))
    pg.draw.circle(screen, (255, 0, 0), (int(vehicle.position[0]), int(vehicle.position[1])), 4)

    v_forward = np.array([vehicle.rad + 1, 0])
    vr_forward = np.empty((2,))
    vr_forward[0] = v_forward[0] * np.cos(vehicle.yaw) - v_forward[1] * np.sin(vehicle.yaw)
    vr_forward[1] = v_forward[0] * np.sin(vehicle.yaw) + v_forward[1] * np.cos(vehicle.yaw)
    m = vehicle.position + vr_forward
    pg.draw.circle(screen, (0, 0, 255), (int(m[0]), int(m[1])), 2)

    frame_elapsed = (dt.datetime.now() - start_frame).microseconds
    print(f'frame took {frame_elapsed}')

    pg.display.flip()
