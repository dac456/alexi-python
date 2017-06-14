import numpy as np
import pygame as pg
from preprocessor import settings


class Vehicle:
    def __init__(self, surf, radius):
       #  self.position = [surf.get_width()/2, surf.get_height()/2]
        self.position = [100, 450]
        self.wheel_speed = [0.0, 0.0]

        self.vel_linear = 2.0
        self.vel_angular = 0.0
        self.tick = 0
        self.rand = np.random.random()

        self.v = 0.0
        self.yaw = 0.0
        self.rad = radius

        self.surface = surf

    def step(self, models, diff, framedata_scaler_in, framedata_scaler_out):
        print(f'tick: {self.tick}')

        pitch = self.__get_pitch()
        roll = self.__get_roll()
        print(f'pitch: {pitch}, roll: {roll}')

        if self.tick < 1000:
            self.vel_linear = self.rand * 3.0
            self.vel_angular = 0
        elif 1000 <= self.tick < 2000:
            self.vel_linear = self.rand * -3.0
            self.vel_angular = 0
        elif 2000 <= self.tick < 3000:
            self.vel_linear = 0.0
            self.vel_angular = self.rand * 1.57
        elif 3000 <= self.tick < 4000:
                self.vel_linear = 0.0
                self.vel_angular = self.rand * -1.57
        elif 4000 <= self.tick < 5000:
            self.vel_linear = self.rand * 3.0
            self.vel_angular = self.rand * 1.57
        elif 5000 <= self.tick < 6000:
            self.vel_linear = self.rand * -3.0
            self.vel_angular = self.rand * -1.57
        elif 6000 <= self.tick < 7000:
            self.vel_linear = self.rand * -3.0
            self.vel_angular = self.rand * 1.57
        elif 7000 <= self.tick < 8000:
            self.vel_linear = self.rand * 3.0
            self.vel_angular = self.rand * -1.57

        if self.tick % 250 == 0:
            self.rand = np.random.random()

        L = 3.2679
        r = 0.5
        self.wheel_speed[0] = (self.vel_linear - (L * self.vel_angular * 0.5)) / r
        self.wheel_speed[1] = (self.vel_linear + (L * self.vel_angular * 0.5)) / r
        print(f'vl: {self.wheel_speed[0]}, vr: {self.wheel_speed[1]}')

        v_in = np.concatenate((self.wheel_speed, [pitch], [roll])).reshape((1, 4))

        sim_terrain = True
        if sim_terrain:
            diff[0, -4:] = v_in
            terrain_pred = models['terrain'].predict(diff)
            terrain_pred = terrain_pred.reshape(settings.crop_dimensions)

            for y in range(terrain_pred.shape[1]):
                for x in range(terrain_pred.shape[0]):
                    offset = terrain_pred[x, y]

                    px = x + int(self.position[0]) - 64
                    py = y + int(self.position[1]) - 64
                    if px < 0:
                        px = 600 + px
                    elif px >= 600:
                        px = px - 600

                    if py < 0:
                        py = 600 + py
                    elif py >= 600:
                        py = py - 600

                    c = self.surface.get_at((px, py))

                    # falloff around vehicle centre
                    """c_pos = np.asarray([x + self.position[0] - 64, y + self.position[1] - 64])
                    d = np.sqrt((c_pos[0] - self.position[0])**2 + (c_pos[1] - self.position[1])**2)
                    # TODO: this should just be 1/d, but that gives various errors
                    offset *= 1.0 - (d / 64.0)"""

                    print(f'offset{offset}')

                    self.surface.set_at((px, py), pg.Color(c.r + int(offset), c.g + int(offset), c.b + int(offset)))

        v_in = framedata_scaler_in.transform(v_in.reshape((1, -1)))

        # dx = models['dx'].predict(v_in)[0][0]
        # dy = models['dy'].predict(v_in)[0][0]
        v = models['v'].predict(v_in)[0][0]
        dtheta = models['dtheta'].predict(v_in)[0][0]
        out = framedata_scaler_out.inverse_transform([[0, 0, v, dtheta]])[0]
        # out = [dx, dy, dtheta]
        # print(f'dx: {out[0]}, dy: {out[1]}, dtheta: {out[2]}')
        # out = [v, dtheta]
        print(f'v: {out[2]}, dtheta: {out[3]}')

        # speed = np.sqrt(out[0]**2 + out[1]**2)
        # if self.vel_linear < 0.0:
        #    speed *= -1.0
        # if self.vel_linear < 0.0:
        #    v *= -1.0

        self.v = out[2]*20.0
        if self.vel_linear < 0.0:
            self.v *= -1.0

        self.yaw += out[3]*0.1
        self.position[0] += self.v * np.cos(self.yaw)
        self.position[1] += self.v * np.sin(self.yaw)

        if self.position[0] >= 600:
            self.position[0] = 0
        elif self.position[0] < 0:
            self.position[0] = 599

        if self.position[1] >= 600:
            self.position[1] = 0
        elif self.position[1] < 0:
            self.position[1] = 599

        self.tick += 1

    def __get_pitch(self):
        max_height = 5.0

        v_forward = np.array([self.rad + 1, 0])
        # r = np.dot(v_forward, self.get_matrix()).astype(int)
        vr = np.empty((2,))
        vr[0] = v_forward[0] * np.cos(self.yaw) - v_forward[1] * np.sin(self.yaw)
        vr[1] = v_forward[0] * np.sin(self.yaw) + v_forward[1] * np.cos(self.yaw)

        d = self.position - vr
        back = self.surface.get_at((int(d[0]), int(d[1])))[0]

        d = self.position + vr
        front = self.surface.get_at((int(d[0]), int(d[1])))[0]

        f = (float(front) / 255.0) * max_height
        b = (float(back) / 255.0) * max_height

        return np.arctan((f - b) / float(self.rad * 2))

    def __get_roll(self):
        max_height = 5.0

        v_right = np.array([0, self.rad + 1])
        # r = np.dot(v_right, self.get_matrix()).astype(int)
        vr = np.empty((2,))
        vr[0] = v_right[0] * np.cos(self.yaw) - v_right[1] * np.sin(self.yaw)
        vr[1] = v_right[0] * np.sin(self.yaw) + v_right[1] * np.cos(self.yaw)

        d = self.position - vr
        left = self.surface.get_at((int(d[0]), int(d[1])))[0]

        d = self.position + vr
        right = self.surface.get_at((int(d[0]), int(d[1])))[0]

        l = (float(left) / 255.0) * max_height
        r = (float(right) / 255.0) * max_height

        return np.arctan((r - l) / float(self.rad * 2))

    def get_matrix(self):
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        return np.array([[c, -s], [s, c]])

