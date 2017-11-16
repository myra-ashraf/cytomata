import pygame as pg
from .settings import *


vec = pg.math.Vector2


class Ally(pg.sprite.Sprite):
    def __init__(self, game, x, y):
        self.groups = game.all_sprites, game.allies
        pg.sprite.Sprite.__init__(self, self.groups)
        self.game = game
        self.image = game.ally_img
        self.rect = self.image.get_rect()
        self.pos = vec(x, y) * TILESIZE
        self.vel = vec(0, 0)
        self.last_update = pg.time.get_ticks()

    def get_keys(self):
        self.vel = vec(0, 0)
        keys = pg.key.get_pressed()
        if keys[pg.K_LEFT] or keys[pg.K_a]:
            self.vel.x = -ALLY_SPEED
        if keys[pg.K_RIGHT] or keys[pg.K_d]:
            self.vel.x = ALLY_SPEED
        if keys[pg.K_UP] or keys[pg.K_w]:
            self.vel.y = -ALLY_SPEED
        if keys[pg.K_DOWN] or keys[pg.K_s]:
            self.vel.y = ALLY_SPEED
        if self.vel.x != 0 and self.vel.y != 0:
            self.vel *= 0.7071

    def collide_wall(self, ax):
        if ax == 'x':
            hits = pg.sprite.spritecollide(self, self.game.walls, False)
            if hits:
                if self.vel.x > 0:
                    self.pos.x = hits[0].rect.left - self.rect.width
                if self.vel.x < 0:
                    self.pos.x = hits[0].rect.right
                self.vel.x = 0
                self.rect.x = self.pos.x
        if ax == 'y':
            hits = pg.sprite.spritecollide(self, self.game.walls, False)
            if hits:
                if self.vel.y > 0:
                    self.pos.y = hits[0].rect.top - self.rect.height
                if self.vel.y < 0:
                    self.pos.y = hits[0].rect.bottom
                self.vel.y = 0
                self.rect.y = self.pos.y

    def collide_enemy(self):
        hits = pg.sprite.spritecollide(self, self.game.mobs, True)
        if hits:
            self.game.eat_snd.play()
            for hit in hits:
                self.game.score += 15

    def time_penalty(self):
        now = pg.time.get_ticks()
        if now - self.last_update > 600:
            self.last_update = now
            self.game.score -= 1

    def update(self):
        self.get_keys()
        self.pos += self.vel * self.game.dt
        self.rect.x = self.pos.x
        self.collide_wall('x')
        self.rect.y = self.pos.y
        self.collide_wall('y')
        self.collide_enemy()
        self.time_penalty()


class Mob(pg.sprite.Sprite):
    def __init__(self, game, x, y):
        self.groups = game.all_sprites, game.mobs
        pg.sprite.Sprite.__init__(self, self.groups)
        self.game = game
        self.image = game.mob_img
        self.rect = self.image.get_rect()
        self.rect.topleft = vec(x, y) * TILESIZE


class Wall(pg.sprite.Sprite):
    def __init__(self, game, x, y):
        self.groups = game.all_sprites, game.walls
        pg.sprite.Sprite.__init__(self, self.groups)
        self.game = game
        self.image = game.wall_img
        self.rect = self.image.get_rect()
        self.x = x
        self.y = y
        self.rect.x = x * TILESIZE
        self.rect.y = y * TILESIZE