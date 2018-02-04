import random as rnd

import numpy as np
import pygame as pg
import pymunk as pm

from .settings import *


class Cell(object):
    """Generic cell template"""
    def __init__(self, game, x, y):
        self.life = 1
        self.mass = 20.0
        self.radius = TILESIZE / 2.1
        self.inertia = pm.moment_for_circle(self.mass, 0, self.radius, (0, 0))
        self.body = pm.Body(self.mass, self.inertia)
        self.body.position = (x + 0.5) * TILESIZE, (y + 0.5) * TILESIZE
        self.body.velocity = pm.Vec2d(0, 0)
        self.shape = pm.Circle(self.body, self.radius, (0, 0))
        self.shape.elasticity = 0.2
        self.shape.friction = 1.0
        self.shape.collision_type = 0
        self.game = game
        self.game.space.add(self.body, self.shape)
        self.game.all_sprites.append(self)
        self.image = pg.Surface((self.radius * 2, self.radius * 2))
        self.image.fill((0, 0, 0, 0))
        self.image.set_colorkey((0, 0, 0))
        pg.draw.circle(self.image, WHITE,
            (int(self.radius), int(self.radius)), int(self.radius), 0)
        self.rect = self.image.get_rect()

    def cap_speed(self, max_speed):
        if self.body.velocity.length > max_speed:
            self.body.velocity = self.body.velocity.normalized() * max_speed

    def apply_friction(self, factor):
        if self.body.velocity.length > 1.0:
            self.body.velocity = self.body.velocity * factor
        else:
            self.body.velocity = (0.0, 0.0)

    def random_walk(self, interval, magnitude):
        interval = max(interval + rnd.randint(-4, 4), 1)
        magnitude += rnd.randint(-20, 20)
        if self.game.ep_step % interval == 0:
            vel = pm.Vec2d(magnitude, 0)
            rand_vel = vel.rotated_degrees(rnd.randint(0, 359))
            self.body.velocity = rand_vel

    def get_tile_position(self):
        x, y = self.body.position
        x = int(np.around((x / TILESIZE) - 0.5))
        y = int(np.around((y / TILESIZE) - 0.5))
        return x, y

    def close_to_edge(self):
        x, y = self.body.position
        return (x < TILESIZE or x > WIDTH - TILESIZE
            or y < TILESIZE or y > HEIGHT - TILESIZE)

    def out_of_view(self):
        x, y = self.body.position
        return (x < 0 or x > WIDTH or y < 0 or y > HEIGHT)

    def selected(self):
        is_selected = False
        clicks = pg.mouse.get_pressed()
        if clicks[0]:  # Left mouse button is pressed
            mouse_pos = pm.Vec2d(pg.mouse.get_pos())
            is_selected = self.rect.collidepoint(mouse_pos)
        return is_selected


class Proxy(Cell):
    """A cell controllable by the agent"""
    def __init__(self, game, x, y):
        self.life = 1
        self.mass = 20.0
        self.radius = TILESIZE / 2.1
        self.inertia = pm.moment_for_circle(self.mass, 0, self.radius, (0, 0))
        self.body = pm.Body(self.mass, self.inertia)
        self.body.position = (x + 0.5) * TILESIZE, (y + 0.5) * TILESIZE
        self.body.velocity = pm.Vec2d(0, 0)
        self.shape = pm.Circle(self.body, self.radius, (0, 0))
        self.shape.elasticity = 0.2
        self.shape.friction = 1.0
        self.shape.collision_type = 1
        self.game = game
        self.game.space.add(self.body, self.shape)
        self.game.all_sprites.append(self)
        self.game.proxies.append(self)
        self.image = self.game.proxy_img
        self.rect = self.image.get_rect()

    def update(self, action):
        self.check_inputs()
        if action is not None:
            self.move(ACTION_MEANING[action])
        self.cap_speed(PROXY_MAX_SPEED)
        self.apply_friction(PROXY_FRICTION)

    def check_inputs(self):
        keys = pg.key.get_pressed()
        if keys[pg.K_LEFT] or keys[pg.K_a]:
            self.move('LEFT')
        if keys[pg.K_RIGHT] or keys[pg.K_d]:
            self.move('RIGHT')
        if keys[pg.K_UP] or keys[pg.K_w]:
            self.move('UP')
        if keys[pg.K_DOWN] or keys[pg.K_s]:
            self.move('DOWN')

    def move(self, direction):
        if direction == 'UP':
            self.body.velocity += (0, PROXY_SPEED)
        if direction == 'DOWN':
            self.body.velocity += (0, -PROXY_SPEED)
        if direction == 'LEFT':
            self.body.velocity += (-PROXY_SPEED, 0)
        if direction == 'RIGHT':
            self.body.velocity += (PROXY_SPEED, 0)

    def penalize(self):
        if self.close_to_edge():
            self.game.reward -= 0.001
        if self.out_of_view():
            self.game.reward -= 0.01

class Cyte(Cell):
    """Passive cell not directly controlled by the agent"""
    def __init__(self, game, x, y):
        self.life = 1
        self.mass = 3000.0
        self.radius = TILESIZE / 2.1
        self.inertia = pm.moment_for_circle(self.mass, 0, self.radius, (0, 0))
        self.body = pm.Body(self.mass, self.inertia)
        self.body.position = (x + 0.5) * TILESIZE, (y + 0.5) * TILESIZE
        self.body.velocity = pm.Vec2d(0, 0)
        self.shape = pm.Circle(self.body, self.radius, (0, 0))
        self.shape.elasticity = 0.2
        self.shape.friction = 10.0
        self.shape.collision_type = 2
        self.game = game
        self.game.space.add(self.body, self.shape)
        self.game.all_sprites.append(self)
        self.game.cytes.append(self)
        self.image = self.game.cyte_img
        self.rect = self.image.get_rect()

    def update(self, action):
        self.apply_friction(0.9)
        # self.decrease_shield(0.1)
        # self.random_walk(100, 20)

    # def decrease_shield(self, interval):
    #     if self.shield and self.game.ep_step % interval == 0:
    #         self.shield -= 1

class Cancer(Cell):
    """Enemy character"""
    def __init__(self, game, x, y):
        self.life = 1
        self.mass = 40.0
        self.radius = TILESIZE / 2.1
        self.inertia = pm.moment_for_circle(self.mass, 0, self.radius, (0, 0))
        self.body = pm.Body(self.mass, self.inertia)
        self.body.position = (x + 0.5) * TILESIZE, (y + 0.5) * TILESIZE
        self.body.velocity = pm.Vec2d(0, 0)
        self.shape = pm.Circle(self.body, self.radius, (0, 0))
        self.shape.elasticity = 0.2
        self.shape.friction = 1.0
        self.shape.collision_type = 3
        self.game = game
        self.game.space.add(self.body, self.shape)
        self.game.all_sprites.append(self)
        self.game.cancers.append(self)
        self.image = self.game.cancer_img
        self.rect = self.image.get_rect()

    def update(self, action):
        self.apply_friction(0.9)
        self.random_walk(12, 20)
        # self.out_of_view(20000)
