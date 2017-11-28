import pygame as pg
from .settings import *


vec = pg.math.Vector2


class Cell(pg.sprite.Sprite):
    def __init__(self, game, x, y):
        self.groups = game.all_sprites, game.cells
        pg.sprite.Sprite.__init__(self, self.groups)
        self.game = game
        self.image = game.cell_img
        self.rect = self.image.get_rect()
        self.x = x
        self.y = y
        self.rect.x = x * TILESIZE
        self.rect.y = y * TILESIZE


class Proxy(pg.sprite.Sprite):
    def __init__(self, game, x, y):
        self.groups = game.all_sprites, game.proxies
        pg.sprite.Sprite.__init__(self, self.groups)
        self.game = game
        self.image = game.proxy_img
        self.rect = self.image.get_rect()
        self.is_selected = False
        self.pos = vec(x, y) * TILESIZE
        self.vel = vec(0, 0)
        self.last_update = pg.time.get_ticks()

    def check_selected(self):
        clicks = pg.mouse.get_pressed()
        if clicks[0]:
            mouse_x, mouse_y = pg.mouse.get_pos()
            mouse_pos = vec(mouse_x, mouse_y)
            proxy_camera_rect = self.game.camera.apply(self)
            if proxy_camera_rect.collidepoint(mouse_x, mouse_y):
                for proxy in self.game.proxies:
                    proxy.is_selected = False
                self.is_selected = True

    def get_inputs(self, control_scheme='joystick'):
        if control_scheme == 'joystick':
            self.vel = vec(0, 0)
            keys = pg.key.get_pressed()
            if keys[pg.K_LEFT] or keys[pg.K_a]:
              self.vel.x = -PROXY_SPEED
            if keys[pg.K_RIGHT] or keys[pg.K_d]:
              self.vel.x = PROXY_SPEED
            if keys[pg.K_UP] or keys[pg.K_w]:
              self.vel.y = -PROXY_SPEED
            if keys[pg.K_DOWN] or keys[pg.K_s]:
              self.vel.y = PROXY_SPEED
            if self.vel.x != 0 and self.vel.y != 0:
              self.vel *= 0.7071
        elif control_scheme == 'pointer':
            self.vel = vec(0, 0)
            clicks = pg.mouse.get_pressed()
            if clicks[0]:
                mouse_x, mouse_y = pg.mouse.get_pos()
                mouse_pos = vec(mouse_x, mouse_y)
                proxy_camera_rect = self.game.camera.apply(self)
                if proxy_camera_rect.collidepoint(mouse_x, mouse_y):
                    mvmt = mouse_pos - proxy_camera_rect.center
                    try:
                        norm_mvmt = mvmt.normalize()
                    except ValueError:
                        norm_mvmt = mvmt
                    self.vel.x = norm_mvmt.x * PROXY_SPEED
                    self.vel.y = norm_mvmt.y * PROXY_SPEED
        elif control_scheme == 'rts':
            pass
        elif control_scheme == 'mask':
            pass
        else:
            raise ValueError('Valid options for control_scheme: joystick, pointer, rts, mask')


    def collide_cell(self, ax):
        if ax == 'x':
            hits = pg.sprite.spritecollide(self, self.game.cells, False)
            if hits:
                if self.vel.x > 0:
                    self.pos.x = hits[0].rect.left - self.rect.width
                if self.vel.x < 0:
                    self.pos.x = hits[0].rect.right
                self.vel.x = 0
                self.rect.x = self.pos.x
        if ax == 'y':
            hits = pg.sprite.spritecollide(self, self.game.cells, False)
            if hits:
                if self.vel.y > 0:
                    self.pos.y = hits[0].rect.top - self.rect.height
                if self.vel.y < 0:
                    self.pos.y = hits[0].rect.bottom
                self.vel.y = 0
                self.rect.y = self.pos.y

    def collide_enemy(self):
        hits = pg.sprite.spritecollide(self, self.game.cancers, True)
        if hits:
            self.game.eat_snd.play()
            for hit in hits:
                self.game.score += 15

    def time_penalty(self):
        now = pg.time.get_ticks()
        if now - self.last_update > 2400:
            self.last_update = now
            self.game.score -= 1

    def update(self):
        self.check_selected()
        if self.is_selected:
            self.get_inputs()
            self.pos += self.vel * self.game.dt
        self.rect.x = self.pos.x
        self.collide_cell('x')
        self.rect.y = self.pos.y
        self.collide_cell('y')
        self.collide_enemy()
        self.time_penalty()


class Cancer(pg.sprite.Sprite):
    def __init__(self, game, x, y):
        self.groups = game.all_sprites, game.cancers
        pg.sprite.Sprite.__init__(self, self.groups)
        self.game = game
        self.image = game.cancer_img
        self.rect = self.image.get_rect()
        self.rect.topleft = vec(x, y) * TILESIZE
