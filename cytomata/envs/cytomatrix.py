import sys
import os
import time
import random as rnd
import pygame as pg
import pymunk as pm
import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from .settings import *
from .sprites import *
from .tilemap import *


class CytomatrixEnv(gym.Env):
    """Main Class"""
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        """Initialize game, window, etc."""
        pg.mixer.pre_init(44100, -16, 2, 2048)
        pg.mixer.init()
        pg.init()
        self.screen = pg.display.set_mode((WIDTH, HEIGHT))
        pg.display.set_caption(TITLE)
        # Clock for setting tempo of game
        self.clock = pg.time.Clock()
        # Time before key hold detected + frequency of key press
        pg.key.set_repeat(100, 100)
        self.load_data()
        self._action_set = ['NO_OP', 'UP', 'DOWN', 'LEFT', 'RIGHT']
        self.action_space = spaces.Discrete(len(self._action_set))
        self.observation_space = spaces.Box(low=0, high=255, shape=(HEIGHT, WIDTH, 3))
        self.reward_range = (-np.inf, np.inf)
        self.viewer = None

    def load_data(self):
        """Load maps, sprites, and other game resources"""
        self.game_dir = os.path.dirname(__file__)
        assets_dir = os.path.join(self.game_dir, 'assets')
        img_dir = os.path.join(assets_dir, 'images')
        snd_dir = os.path.join(assets_dir, 'sounds')
        music_dir = os.path.join(assets_dir, 'music')
        self.map = Map(os.path.join(assets_dir, 'maps', MAP_FILE))
        self.bkg_img = pg.image.load(os.path.join(img_dir, 'bkgd', BKG_IMG)).convert()
        self.bkg_img = pg.transform.scale(self.bkg_img, (WIDTH, HEIGHT))
        self.bkg_rect = self.bkg_img.get_rect()
        self.proxy_img = pg.image.load(os.path.join(img_dir, 'chars', PROXY_IMG)).convert_alpha()
        self.proxy_img = pg.transform.scale(self.proxy_img, (TILESIZE, TILESIZE))
        self.cyte_img = pg.image.load(os.path.join(img_dir, 'chars', CYTE_IMG)).convert_alpha()
        self.cyte_img = pg.transform.scale(self.cyte_img, (TILESIZE, TILESIZE))
        self.cancer_img = pg.image.load(os.path.join(img_dir, 'chars', CANCER_IMG)).convert_alpha()
        self.cancer_img = pg.transform.scale(self.cancer_img, (TILESIZE, TILESIZE))
        self.eat_snd = pg.mixer.Sound(os.path.join(snd_dir, EAT_SND))
        # pg.mixer.music.load(os.path.join(music_dir, MUSIC))
        # pg.mixer.music.set_volume(1.0)

    def new(self):
        """Set up objects (sprites, camera, etc.) for new game"""
        self.terminal = False
        self.space = pm.Space()
        self.space.gravity = (0.0, 0.0)
        self.space.add_collision_handler(0, 2).post_solve = self.proxy_cancer_collision
        self.space.add_collision_handler(0, 1).post_solve = self.proxy_cyte_collision
        self.all_sprites = []
        self.proxies = []
        self.cytes = []
        self.cancers = []
        self.map.occupied_tiles = []
        self.score = 0.0
        self.timer = 0.0
        self.timer_start = time.time()
        self.last_update = pg.time.get_ticks()
        self.spawn_from_map()
        self.spawn_randomly(Cancer, NUM_RANDOM_CANCERS, 'center0', 16)
        self.spawn_randomly(Cyte, NUM_RANDOM_CYTES)
        self.spawn_randomly(Proxy, NUM_RANDOM_PROXIES, 'center1')
        # self.camera = Camera(self.map.width, self.map.height)
        # pg.mixer.music.play(-1)

    def events(self):
        """Game loop - process inputs/events"""
        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                self.quit()

    def update(self, action):
        """Game loop - updates"""
        self.timer = time.time() - self.timer_start
        if self.timer > 300.0:
            self.terminal = True
        for sprite in self.all_sprites:
            if sprite in self.proxies:
                sprite.update(action)
            else:
                sprite.update()
        self.time_penalty(2000)
        # Camera tracking
        # self.camera.update(self.proxy)
        # End the current game if all cancers have been eliminated
        if len(self.cancers) < 1:
            self.terminal = True

    def draw(self):
        """Game loop - render"""
        if DEBUG:
            pg.display.set_caption('{:.2f}'.format(self.clock.get_fps()))
        self.screen.fill(BGCOLOR)
        self.screen.blit(self.bkg_img, self.bkg_rect)
        # self.draw_grid()
        for sprite in self.all_sprites:
            self.draw_sprite(self.screen, sprite, sprite.image)
        self.draw_text(self.screen, 'Score: {:.2f}'.format(self.score), 18, WIDTH * 0.9, 8)
        self.draw_text(self.screen, 'Time: ' + str('{:.2f}'.format(self.timer)), 18, WIDTH * 0.1, 8)
        self.space.step(1.0 / FPS)
        self.raw_img = pg.surfarray.array3d(pg.display.get_surface())
        pg.display.flip()

    def _reset(self):
        self.new()
        for i in range(4):
            self.step(None)
        self.score = 0.0
        first_img, _, _, _ = self.step(None)
        return first_img

    def _step(self, action):
        """Game loop"""
        self.events()
        self.update(action)
        self.draw()
        self.clock.tick(FPS)
        return self.raw_img, self.score, self.terminal, {}

    def quit(self):
        """Quit to desktop"""
        pg.quit()
        sys.exit()

    def spawn_from_map(self):
        """Spawn objects based on locations specified in the map file"""
        for row, tiles in enumerate(self.map.data):
            for col, tile in enumerate(tiles):
                # Convert coords to match y-axis position as seen on map
                map_row = int(GRIDHEIGHT - row - 1.0)
                if tile == '0':
                    self.map.occupied_tiles.append((col, map_row))
                    Cyte(self, col, map_row)
                if tile == '1':
                    self.map.occupied_tiles.append((col, map_row))
                    Proxy(self, col, map_row)
                if tile == '2':
                    self.map.occupied_tiles.append((col, map_row))
                    Cancer(self, col, map_row)

    def get_open_spots(self, min_x, max_x, min_y, max_y):
        """Return grid coordinates of locations on the map not occupied by some object"""
        open_spots = [
            (x, y) for x in range(min_x, max_x)
            for y in range(min_y, max_y)
            if (x, y) not in self.map.occupied_tiles
        ]
        return open_spots

    def spawn_randomly(self, Entity, number, bias=None, rand_buffer=0):
        """Creates a proxy/cyte/cancer in a random spot on the map"""
        if bias == 'center0':
            open_spots = self.get_open_spots(1, int(self.map.tile_width - 1), 1, int(self.map.tile_height - 1))
        elif bias == 'center1':
            open_spots = self.get_open_spots(7, int(self.map.tile_width - 7), 5, int(self.map.tile_height - 5))
        else:
            open_spots = self.get_open_spots(0, int(self.map.tile_width), 0, int(self.map.tile_height))
        number += rnd.randint(-rand_buffer, rand_buffer)
        for i in range(number):
            rand_x, rand_y = rnd.choice(open_spots)
            Entity(self, rand_x, rand_y)
            open_spots.remove((rand_x, rand_y))
            self.map.occupied_tiles.append((rand_x, rand_y))

    def draw_sprite(self, screen, Entity, image):
        p = pm.Vec2d(self.to_pygame(*Entity.body.position))
        # pg.draw.circle(screen, (0,0,255), p, int(Entity.radius), 2)
        img_offset = (20.0, 20.0)
        p -= img_offset
        screen.blit(image, p)

    def draw_grid(self):
        """Draw the tiles based on settings"""
        for x in range(0, WIDTH, TILESIZE):
            pg.draw.line(self.screen, LIGHTGREY, (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, TILESIZE):
            pg.draw.line(self.screen, LIGHTGREY, (0, y), (WIDTH, y))

    def draw_text(self, surf, text, size, x, y):
        """Draws text on a specified surface"""
        font_name = pg.font.match_font('arial')
        font = pg.font.Font(font_name, size)
        text_surface = font.render(text, True, DARKGREY)
        text_rect = text_surface.get_rect()
        text_rect.midtop = (x, y)
        surf.blit(text_surface, text_rect)

    def to_pygame(self, x, y):
        """Convert pymunk to pygame coordinates"""
        return int(x), int(-y + HEIGHT)

    def proxy_cancer_collision(self, arbiter, space, _):
        """Collision between bird and pig"""
        a, b = arbiter.shapes
        cancer_body = b.body
        for cancer in self.cancers:
            if cancer_body == cancer.body:
                self.score += 1.0
                self.space.remove(cancer.shape, cancer.shape.body)
                self.cancers.remove(cancer)
                self.all_sprites.remove(cancer)
                self.eat_snd.play()

    def proxy_cyte_collision(self, arbiter, space, _):
        """Collision between bird and pig"""
        a, b = arbiter.shapes
        cyte_body = b.body
        for cyte in self.cytes:
            if cyte_body == cyte.body and not cyte.shield:
                cyte.life -= 1
                if cyte.life <= 0:
                    self.space.remove(cyte.shape, cyte.shape.body)
                    self.cytes.remove(cyte)
                    self.all_sprites.remove(cyte)
                    self.score -= 0.1
                else:
                    cyte.shield = 12

    def time_penalty(self, duration):
        now = pg.time.get_ticks()
        if now - self.last_update > duration:
            self.last_update = now
            self.score -= 0.05
