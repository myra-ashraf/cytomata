import sys
import os
import random as rnd
import pygame as pg
import pymunk as pm
import numpy as np
import gym
from gym import spaces
from .settings import *
from .sprites import *
from .tilemap import *


class CytomatrixEnv(gym.Env):
    """Main Class"""
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        """Initialize game, window, etc."""
        pg.mixer.pre_init(22050, -16, 2, 512)
        pg.init()
        self.screen = pg.display.set_mode((WIDTH, HEIGHT))
        pg.display.set_caption(TITLE)
        self.clock = pg.time.Clock()
        # Time before key hold detected + frequency of key press
        pg.key.set_repeat(100, 100)
        self.load_data()
        # Required for gym env
        self._action_set = range(0, len(ACTION_MEANING))
        self.action_space = spaces.Discrete(len(self._action_set))
        self.observation_space = spaces.Box(low=0, high=255, shape=(HEIGHT, WIDTH, 3))
        self.reward_range = (-np.inf, np.inf)
        self.viewer = None

    def load_data(self):
        """Load maps, sprites, and other game resources"""
        self.game_dir = os.path.dirname(__file__)
        assets_dir = os.path.join(self.game_dir, 'assets')
        img_dir = os.path.join(self.game_dir, 'assets', 'images')
        snd_dir = os.path.join(self.game_dir, 'assets', 'sounds')
        music_dir = os.path.join(self.game_dir, 'assets', 'music')
        self.map = Map(os.path.join(self.game_dir, 'assets', 'maps', MAP_FILE))
        self.bkg_img = pg.image.load(os.path.join(img_dir, 'bkgd', BKG_IMG)).convert()
        self.bkg_img = pg.transform.scale(self.bkg_img, (WIDTH, HEIGHT))
        self.proxy_img = pg.image.load(os.path.join(img_dir, 'chars', PROXY_IMG)).convert_alpha()
        self.proxy_img = pg.transform.scale(self.proxy_img, (TILESIZE, TILESIZE))
        self.cyte_img = pg.image.load(os.path.join(img_dir, 'chars', CYTE_IMG)).convert_alpha()
        self.cyte_img = pg.transform.scale(self.cyte_img, (TILESIZE, TILESIZE))
        self.cancer_img = pg.image.load(os.path.join(img_dir, 'chars', CANCER_IMG)).convert_alpha()
        self.cancer_img = pg.transform.scale(self.cancer_img, (TILESIZE, TILESIZE))
        # self.eat_snd = pg.mixer.Sound(os.path.join(snd_dir, EAT_SND))
        # pg.mixer.music.load(os.path.join(music_dir, MUSIC))
        # pg.mixer.music.set_volume(1.0)

    def new(self):
        """Set up objects (sprites, camera, etc.) for new game"""
        self.terminal = False
        self.space = pm.Space()
        self.space.gravity = (0.0, 0.0)
        self.space.add_collision_handler(1, 3).post_solve = self.proxy_cancer_collision
        self.space.add_collision_handler(1, 2).post_solve = self.proxy_cyte_collision
        self.all_sprites = []
        self.proxies = []
        self.cytes = []
        self.cancers = []
        self.score = 0.0
        self.reward = 0.0
        self.timer = 0.0
        self.last_update = pg.time.get_ticks()
        self.boundary_box()
        self.spawn_from_map()
        self.spawn_randomly(Proxy, NUM_RANDOM_PROXIES)
        self.spawn_randomly(Cancer, NUM_RANDOM_CANCERS, spaced=True)
        self.spawn_randomly(Cyte, NUM_RANDOM_CYTES)
        # self.camera = Camera(self.map.width, self.map.height)
        # pg.mixer.music.play(-1)

    def boundary_box(self):
        """Create physical borders around the display edges"""
        self.static_lines = [
            # LEFT
            pm.Segment(self.space.static_body, (0, 0), (0, HEIGHT), 1),
            # TOP
            pm.Segment(self.space.static_body, (0, HEIGHT), (WIDTH, HEIGHT), 1),
            # RIGHT
            pm.Segment(self.space.static_body, (WIDTH, HEIGHT), (WIDTH, 0), 1),
            # BOTTOM
            pm.Segment(self.space.static_body, (WIDTH, 0), (0, 0), 1)
        ]
        for line in self.static_lines:
            line.elasticity = 0.1
            line.collision_type = 0
        self.space.add(self.static_lines)

    def spawn_from_map(self):
        """Spawn objects based on locations specified in the map file"""
        for row, tiles in enumerate(self.map.data):
            for col, tile in enumerate(tiles):
                # Convert coords to match y-axis position as seen on map
                map_row = int(GRIDHEIGHT - row - 1.0)
                if tile == '0':
                    Cyte(self, col, map_row)
                if tile == '1':
                    Proxy(self, col, map_row)
                if tile == '2':
                    Cancer(self, col, map_row)

    def spawn_randomly(self, Entity, number, rand_num=0, bias=None, spaced=False):
        """Creates a proxy/cyte/cancer in a random spot on the map"""
        if bias == 'center':
            open_spots = self.get_open_spots(
                1, int(self.map.tile_width - 1),
                1, int(self.map.tile_height - 1),
                spaced=spaced)
        else:
            open_spots = self.get_open_spots(
                0, int(self.map.tile_width),
                0, int(self.map.tile_height),
                spaced=spaced)
        number += rnd.randint(-rand_num, rand_num)
        for i in range(number):
            rand_x, rand_y = rnd.choice(open_spots)
            Entity(self, rand_x, rand_y)
            open_spots.remove((rand_x, rand_y))

    def get_open_spots(self, min_x, max_x, min_y, max_y, spaced=False):
        """Return grid coordinates of locations on the map not occupied by some object"""
        occupied_tiles = self.get_occupied_tiles(spaced=spaced)
        open_spots = [
            (x, y) for x in range(min_x, max_x)
            for y in range(min_y, max_y)
            if (x, y) not in occupied_tiles]
        return open_spots

    def get_occupied_tiles(self, spaced=False):
        occupied_tiles = []
        for sprite in self.all_sprites:
            tile_position = sprite.get_tile_position()
            # Spawn this cell at least 1 spot away from a proxy
            if spaced and sprite in self.proxies:
                x, y = tile_position
                x_coords = [x - 1, x, x + 1]
                y_coords = [y - 1, y, y + 1]
                invalidated = [(x, y) for x in x_coords for y in y_coords]
                occupied_tiles += invalidated
                print(occupied_tiles)
            else:
                occupied_tiles.append(tile_position)
        return occupied_tiles

    def events(self):
        """Game loop - process inputs/events"""
        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                self.quit()

    def update(self, action):
        """Game loop - updates"""
        self.timer += self.clock.get_time() / 1000.0
        if self.timer > MAX_TIME:
            self.terminal = True
        for sprite in self.all_sprites:
            if sprite in self.proxies:
                sprite.update(action)
            else:
                sprite.update()
        # self.time_penalty(2000)
        # Camera tracking
        # self.camera.update(self.proxy)
        if len(self.cancers) < 1:
            # self.terminal = True
            self.spawn_randomly(Cancer, 1)

    def _reset(self):
        self.new()
        first_img, _, _, _ = self.step(0)
        return first_img

    def _step(self, action):
        """Game loop"""
        self.score0 = self.score
        self.reward = 0.0
        self.events()
        self.update(action)
        self.draw()
        self.clock.tick(FPS)
        self.reward += self.score - self.score0
        return self.get_raw_img(), np.around(self.reward, 2), self.terminal, {}

    def draw(self):
        """Game loop - render"""
        self.screen.fill(WHITE)
        for sprite in self.all_sprites:
            self.draw_sprite(self.screen, sprite, sprite.image)
        fps_str = 'FPS: {:.2f}'.format(self.clock.get_fps())
        score_str = ' | Score: {:.2f}'.format(self.score)
        time_str = ' | Time: {:.2f}'.format(self.timer)
        pg.display.set_caption(fps_str + score_str + time_str)
        # for line in self.static_lines:
        #     p1 = pm.Vec2d(self.to_pygame(*line.a))
        #     p2 = pm.Vec2d(self.to_pygame(*line.b))
        #     pg.draw.lines(self.screen, BLACK, False, [p1, p2])
        # self.screen.blit(self.bkg_img, self.bkg_img.get_rect())
        # self.draw_grid()
        # self.draw_text(self.screen, 'Score: {:.2f}'.format(self.score), 18, WIDTH * 0.9, 8)
        # self.draw_text(self.screen, 'Time: ' + str('{:.2f}'.format(self.timer)), 18, WIDTH * 0.1, 8)
        for i in range(10):
            self.space.step(GAME_SPEED/FPS/10.0)
        pg.display.flip()

    def get_raw_img(self):
        raw_display_surf = pg.transform.flip(pg.transform.rotate(pg.display.get_surface(), 90), False, True)
        return pg.surfarray.array3d(raw_display_surf)

    def draw_sprite(self, screen, Entity, image):
        p = pm.Vec2d(self.to_pygame(*Entity.body.position))
        # pg.draw.circle(screen, (0,0,255), p, int(Entity.radius), 2)
        img_offset = (TILESIZE / 2, TILESIZE / 2)
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
        a, b = arbiter.shapes
        cancer_body = b.body
        for cancer in self.cancers:
            if cancer_body == cancer.body:
                self.score += 1.0
                self.space.remove(cancer.shape, cancer.shape.body)
                self.cancers.remove(cancer)
                self.all_sprites.remove(cancer)
                # self.eat_snd.play()

    def proxy_cyte_collision(self, arbiter, space, _):
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

    def get_keys_to_action(self):
        KEYWORD_TO_KEY = {
            'UP': ord('w'),
            'DOWN': ord('s'),
            'LEFT': ord('a'),
            'RIGHT': ord('d')}
        keys_to_action = {}
        for action_id, action_meaning in enumerate(self.get_action_meanings()):
            keys = []
            for keyword, key in KEYWORD_TO_KEY.items():
                if keyword in action_meaning:
                    keys.append(key)
            keys = tuple(sorted(keys))
            assert keys not in keys_to_action
            keys_to_action[keys] = action_id
        return keys_to_action

    def get_action_meanings(self):
        return [ACTION_MEANING[i] for i in self._action_set]

    def time_penalty(self, duration):
        now = pg.time.get_ticks()
        if now - self.last_update > duration:
            self.last_update = now
            self.score -= 0.01

    def quit(self):
        """Quit to desktop"""
        pg.quit()
        sys.exit()
