import random as rnd
import numpy as np
import pygame as pg
import pymunk as pm
from .settings import *


class Proxy():
    def __init__(self, game, x, y):
        self.mass = 30.0
        self.radius = TILESIZE / 2.0
        self.inertia = pm.moment_for_circle(self.mass, 0, self.radius, (0, 0))
        self.body = pm.Body(self.mass, self.inertia)
        self.body.position = (x + 0.5) * TILESIZE, (y + 0.5) * TILESIZE
        self.shape = pm.Circle(self.body, self.radius, (0, 0))
        self.shape.elasticity = 0.2
        self.shape.friction = 1.0
        self.shape.collision_type = 0
        self.body.velocity = pm.Vec2d(0, 0)
        self.life = 20
        self.last_update0 = pg.time.get_ticks()
        self.last_update1 = pg.time.get_ticks()
        self.game = game
        self.game.space.add(self.body, self.shape)
        self.game.all_sprites.append(self)
        self.game.proxies.append(self)
        self.image = self.game.proxy_img

    def move(self, direction):
        """ Moves the proxy in a direction: UP, DOWN, LEFT, RIGHT"""
        if direction == 'UP':
            self.body.velocity += (0, 30)
        if direction == 'DOWN':
            self.body.velocity += (0, -30)
        if direction == 'LEFT':
            self.body.velocity += (-30, 0)
        if direction == 'RIGHT':
            self.body.velocity += (30, 0)

    def cap_speed(self, max_speed):
        if self.body.velocity.length > max_speed:
            self.body.velocity = self.body.velocity.normalized() * max_speed

    def bkg_friction(self, factor):
        if self.body.velocity.length > 1.0:
            self.body.velocity = self.body.velocity * factor
        else:
            self.body.velocity = (0.0, 0.0)

    def close_to_edge(self, duration):
        now = pg.time.get_ticks()
        x, y = self.body.position
        edging = x < TILESIZE or x > WIDTH - TILESIZE or y < TILESIZE or y > HEIGHT - TILESIZE
        if now - self.last_update0 > duration and edging:
            self.last_update0 = now
            self.game.score -= 0.2

    def out_of_arena(self, duration):
        now = pg.time.get_ticks()
        x, y = self.body.position
        out_of_view = x < 0 or x > WIDTH or y < 0 or y > HEIGHT
        if now - self.last_update1 > duration and out_of_view:
            self.last_update1 = now
            self.game.score -= 1.0
            self.game.terminal = True

    # def check_selected(self):
    #     clicks = pg.mouse.get_pressed()
    #     if clicks[0]:
    #         mouse_x, mouse_y = pg.mouse.get_pos()
    #         mouse_pos = vec(mouse_x, mouse_y)
    #         proxy_camera_rect = self.game.camera.apply(self)
    #         if proxy_camera_rect.collidepoint(mouse_x, mouse_y):
    #         if self.rect.collidepoint(mouse_x, mouse_y):
    #             for proxy in self.game.proxies:
    #                 proxy.is_selected = False
    #             self.is_selected = True

    def check_inputs(self, control_scheme='joystick'):
        if control_scheme == 'joystick':
            keys = pg.key.get_pressed()
            if keys[pg.K_LEFT] or keys[pg.K_a]:
                self.move('LEFT')
            if keys[pg.K_RIGHT] or keys[pg.K_d]:
                self.move('RIGHT')
            if keys[pg.K_UP] or keys[pg.K_w]:
                self.move('UP')
            if keys[pg.K_DOWN] or keys[pg.K_s]:
                self.move('DOWN')
        # elif control_scheme == 'pointer':
        #     self.vel = vec(0, 0)
        #     clicks = pg.mouse.get_pressed()
        #     if clicks[0]:
        #         mouse_x, mouse_y = pg.mouse.get_pos()
        #         mouse_pos = vec(mouse_x, mouse_y)
        #         # proxy_camera_rect = self.game.camera.apply(self)
        #         # if proxy_camera_rect.collidepoint(mouse_x, mouse_y):
        #         if self.rect.collidepoint(mouse_x, mouse_y):
        #             # mvmt = mouse_pos - proxy_camera_rect.center
        #             mvmt = mouse_pos - self.rect.center
        #             try:
        #                 norm_mvmt = mvmt.normalize()
        #             except ValueError:
        #                 norm_mvmt = mvmt
        #             self.vel.x = norm_mvmt.x * PROXY_SPEED
        #             self.vel.y = norm_mvmt.y * PROXY_SPEED
        # elif control_scheme == 'rts':
        #     pass
        # elif control_scheme == 'mask':
        #     pass
        else:
            raise ValueError('Valid options for control_scheme: joystick, pointer, rts, mask')

    def agent_act(self, action):
        actions_ref = ['NO_OP', 'UP', 'DOWN', 'LEFT', 'RIGHT']
        self.move(actions_ref[action])

    def update(self, action):
        self.check_inputs()
        if action is not None:
            self.agent_act(action)
        self.cap_speed(PROXY_SPEED)
        self.bkg_friction(0.9)
        # self.close_to_edge(1000)
        # self.out_of_arena(20000)
        # self.check_selected()
        # if self.is_selected:
        #     self.check_inputs()
        #     self.pos += self.vel * self.game.dt
        # self.time_penalty()


class Cyte():
    def __init__(self, game, x, y):
        self.mass = 300.0
        self.radius = TILESIZE / 2.0
        self.inertia = pm.moment_for_circle(self.mass, 0, self.radius, (0, 0))
        self.body = pm.Body(self.mass, self.inertia)
        self.body.position = (x + 0.5) * TILESIZE, (y + 0.5) * TILESIZE
        self.shape = pm.Circle(self.body, self.radius, (0, 0))
        self.shape.elasticity = 0.2
        self.shape.friction = 10.0
        self.shape.collision_type = 1
        self.life = 5
        self.shield = 0
        self.last_update = pg.time.get_ticks()
        self.rand_walk_t0 = pg.time.get_ticks()
        self.game = game
        self.game.space.add(self.body, self.shape)
        self.game.all_sprites.append(self)
        self.game.cytes.append(self)
        self.image = self.game.cyte_img

    def bkg_friction(self, factor):
        if self.body.velocity.length > 1.0:
            self.body.velocity = self.body.velocity * factor
        else:
            self.body.velocity = (0.0, 0.0)

    def shield_timer(self, duration):
        now = pg.time.get_ticks()
        if self.shield and now - self.last_update > duration:
            self.last_update = now
            self.shield -= 1

    def random_walk(self, delay, magnitude):
        now = pg.time.get_ticks()
        delay += rnd.randint(-800, 800)
        magnitude += rnd.randint(-20, 20)
        if now - self.rand_walk_t0 > delay:
            self.rand_walk_t0 = now
            vel = pm.Vec2d(magnitude, 0)
            rand_angle = rnd.randint(0, 359)
            rand_vel = vel.rotated_degrees(rand_angle)
            self.body.velocity = rand_vel

    def update(self):
        self.bkg_friction(0.9)
        self.shield_timer(100)
        self.random_walk(2200, 20)


class Cancer():
    def __init__(self, game, x, y):
        self.mass = 40.0
        self.radius = TILESIZE / 2.0
        self.inertia = pm.moment_for_circle(self.mass, 0, self.radius, (0, 0))
        self.body = pm.Body(self.mass, self.inertia)
        self.body.position = (x + 0.5) * TILESIZE, (y + 0.5) * TILESIZE
        self.shape = pm.Circle(self.body, self.radius, (0, 0))
        self.shape.elasticity = 0.2
        self.shape.friction = 1.5
        self.shape.collision_type = 2
        self.life = 20
        self.game = game
        self.last_update = pg.time.get_ticks()
        self.rand_walk_t0 = pg.time.get_ticks()
        self.game.space.add(self.body, self.shape)
        self.game.all_sprites.append(self)
        self.game.cancers.append(self)
        self.image = self.game.cancer_img

    def bkg_friction(self, factor):
        if self.body.velocity.length > 1.0:
            self.body.velocity = self.body.velocity * factor
        else:
            self.body.velocity = (0.0, 0.0)

    def out_of_arena(self, duration):
        now = pg.time.get_ticks()
        x, y = self.body.position
        out_of_view = x < 0 or x > WIDTH or y < 0 or y > HEIGHT
        if now - self.last_update > duration and out_of_view:
            self.last_update = now
            self.game.cancers.remove(self)
            self.game.all_sprites.remove(self)

    def random_walk(self, delay, magnitude):
        now = pg.time.get_ticks()
        delay += rnd.randint(-500, 500)
        magnitude += rnd.randint(-20, 20)
        if now - self.rand_walk_t0 > delay:
            self.rand_walk_t0 = now
            vel = pm.Vec2d(magnitude, 0)
            rand_angle = rnd.randint(0, 359)
            rand_vel = vel.rotated_degrees(rand_angle)
            self.body.velocity = rand_vel

    def update(self):
        self.bkg_friction(0.9)
        self.random_walk(1200, 30)
        # self.out_of_arena(20000)
