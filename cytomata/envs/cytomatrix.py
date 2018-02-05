import os
import sys

import gym
from gym import spaces
import numpy as np
import pygame as pg
import pymunk as pm

from .sprites import *
from .settings import *


class Cytomatrix(gym.Env):
    """An environment that mimics microscope image data.
    An agent can be trained using this environment before being applied to
    the real setting.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        if not DISPLAY_SCREEN:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        pg.init()
        pg.mixer.quit()
        self.screen = pg.display.set_mode((WIDTH, HEIGHT), 0, 32)
        pg.display.set_caption(TITLE)
        self.clock = pg.time.Clock()
        self.rng = np.random.RandomState()
        self.load_data()
        self._action_set = range(0, len(ACTION_MEANING))
        self.action_space = spaces.Discrete(len(self._action_set))
        self.observation_space = spaces.Box(low=0, high=255, shape=(WIDTH, HEIGHT, 3), dtype=np.uint8)
        self.reward_range = (-np.inf, np.inf)
        self.viewer = None

    def load_data(self):
        """Load game resources"""
        game_dir = os.path.dirname(__file__)
        proxy_path = os.path.join(game_dir, 'assets', 'images', PROXY_IMG)
        cyte_path = os.path.join(game_dir, 'assets', 'images', CYTE_IMG)
        cancer_path = os.path.join(game_dir, 'assets', 'images', CANCER_IMG)
        self.proxy_img = self.load_and_resize(proxy_path, TILESIZE, TILESIZE)
        self.cyte_img = self.load_and_resize(cyte_path, TILESIZE, TILESIZE)
        self.cancer_img = self.load_and_resize(cancer_path, TILESIZE, TILESIZE)

    def load_and_resize(self, img_path, width, height):
        unscaled = pg.image.load(img_path).convert_alpha()
        return pg.transform.scale(unscaled, (width, height))

    def _reset(self):
        self.terminal = False
        self.ep_step = 0
        self.ep_reward = 0.0
        self.reward = 0.0
        self.timer = 0.0
        self.reset_space()
        self.reset_sprites()
        self.draw()
        return self.get_raw_img()

    def reset_space(self):
        self.space = pm.Space()
        self.space.gravity = (0.0, 0.0)
        self.space.add_collision_handler(1, 3).post_solve = self.proxy_cancer_collision
        # self.space.add_collision_handler(1, 2).post_solve = self.proxy_cyte_collision

    def reset_sprites(self):
        self.all_sprites = []
        self.proxies = []
        self.cytes = []
        self.cancers = []
        self.boundary_box()
        # cyte_static_pos = [
        #     (2, 4), (2, 5), (2, 6), (2, 7), (3, 7),
        #     (4, 7), (5, 7), (6, 7), (6, 6), (6, 5),
        #     (7, 5), (8, 5), (9, 5), (2, 3), (2, 2),
        #     (3, 2), (4, 2), (5, 2), (6, 2), (7, 2)
        # ]
        # for x, y in cyte_static_pos:
        #     Cyte(self, x, y)
        self.spawn_randomly(Proxy, NUM_RANDOM_PROXIES)
        self.spawn_randomly(Cancer, NUM_RANDOM_CANCERS, spaced=True)
        self.spawn_randomly(Cyte, NUM_RANDOM_CYTES)

    def proxy_cancer_collision(self, arbiter, space, _):
        a, b = arbiter.shapes
        cancer_body = b.body
        for cancer in self.cancers:
            if cancer_body == cancer.body:
                self.reward += 1.0
                self.ep_reward += 1.0
                self.space.remove(cancer.shape, cancer.shape.body)
                self.cancers.remove(cancer)
                self.all_sprites.remove(cancer)

    def proxy_cyte_collision(self, arbiter, space, _):
        pass
        # a, b = arbiter.shapes
        # cyte_body = b.body
        # for cyte in self.cytes:
        #     if cyte_body == cyte.body and not cyte.shield:
        #         cyte.life -= 1
        #         if cyte.life <= 0:
        #             self.space.remove(cyte.shape, cyte.shape.body)
        #             self.cytes.remove(cyte)
        #             self.all_sprites.remove(cyte)
        #             self.reward -= 0.1
        #             self.score -= 0.1
        #         else:
        #             cyte.shield = 12

    def boundary_box(self):
        """Create physical borders around the display edges"""
        self.static_lines = [  # left - right - top - bottom
            pm.Segment(self.space.static_body, (0, 0), (0, HEIGHT), 1),
            pm.Segment(self.space.static_body, (0, HEIGHT), (WIDTH, HEIGHT), 1),
            pm.Segment(self.space.static_body, (WIDTH, HEIGHT), (WIDTH, 0), 1),
            pm.Segment(self.space.static_body, (WIDTH, 0), (0, 0), 1)
        ]
        for line in self.static_lines:
            line.elasticity = 0.1
            line.collision_type = 0
        self.space.add(self.static_lines)

    def spawn_randomly(self, Entity, number, rand_num=0, spaced=False):
        """Creates a proxy/cyte/cancer in a random spot on the map"""
        open_spots = self.get_open_spots(
            0, int(WIDTH // TILESIZE),
            0, int(HEIGHT // TILESIZE),
            spaced=spaced)
        number += self.rng.randint(-rand_num, rand_num + 1)
        for i in range(number):
            rand_x, rand_y = open_spots[self.rng.choice(len(open_spots))]
            Entity(self, rand_x, rand_y)
            open_spots.remove((rand_x, rand_y))

    def get_open_spots(self, min_x, max_x, min_y, max_y, spaced=False):
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
            else:
                occupied_tiles.append(tile_position)
        return occupied_tiles

    def _step(self, action):
        self.reward = 0.0
        self.ep_step += 1
        self.events()
        self.update(action)
        self.draw()
        for _ in range(10):
            self.space.step(GAME_SPEED/10)
        self.clock.tick()
        return self.get_raw_img(), self.reward, self.terminal, {}

    def events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT or (
                event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                pg.quit()
                sys.exit()

    def update(self, action):
        self.timer += self.clock.get_time() / 1000.0
        # self.terminal = self.timer > MAX_TIME
        self.terminal = self.ep_step == TERMINAL_STEP
        for sprite in self.all_sprites:
            sprite.update(action)
        if len(self.cancers) < 1:
            self.spawn_randomly(Cancer, NUM_RANDOM_CANCERS, spaced=True)

    def draw(self):
        if not DISPLAY_SCREEN:
            self.screen = pg.Surface((WIDTH, HEIGHT), pg.SRCALPHA, 32)
        self.screen.fill(BLACK)
        for sprite in self.all_sprites:
            self.draw_sprite(self.screen, sprite, sprite.image)
        if SHOW_STATS:
            self.show_stats()
        pg.display.update()

    def show_stats(self):
        fps_str = 'FPS: {:.2f}'.format(self.clock.get_fps())
        score_str = ' | Score: {:.2f}'.format(self.ep_reward)
        time_str = ' | Time: {:.2f}'.format(self.timer)
        pg.display.set_caption(fps_str + score_str + time_str)

    def get_raw_img(self):
        screen_surface = pg.transform.flip(
            pg.transform.rotate(self.screen, 90), False, True
        )
        return pg.surfarray.array3d(screen_surface)

    def draw_sprite(self, screen, Entity, image):
        p = pm.Vec2d(self.to_pygame(*Entity.body.position))
        # pg.draw.circle(screen, (0, 0, 255), p, int(Entity.radius), 2)
        img_offset = (TILESIZE / 2, TILESIZE / 2)
        p -= img_offset
        screen.blit(image, p)

    def draw_borders(self):
        for line in self.static_lines:
            p1 = pm.Vec2d(self.to_pygame(*line.a))
            p2 = pm.Vec2d(self.to_pygame(*line.b))
            pg.draw.lines(self.screen, (255, 0, 0), False, [p1, p2])

    def to_pygame(self, x, y):
        """Convert pymunk to pygame coordinates"""
        return int(x), int(HEIGHT - y)

    def draw_grid(self):
        """Draw grid lines on screen"""
        for x in range(0, WIDTH, TILESIZE):
            pg.draw.line(self.screen, LIGHTGREY, (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, TILESIZE):
            pg.draw.line(self.screen, LIGHTGREY, (0, y), (WIDTH, y))

    def draw_text(self, surf, text, size, x, y):
        """Draw text on a specified surface"""
        font_name = pg.font.match_font('arial')
        font = pg.font.Font(font_name, size)
        text_surface = font.render(text, True, DARKGREY)
        text_rect = text_surface.get_rect()
        text_rect.midtop = (x, y)
        surf.blit(text_surface, text_rect)

    def get_action_meanings(self):
        return [ACTION_MEANING[i] for i in self._action_set]

    def _seed(self, seed):
        self.rng = np.random.RandomState(seed)

    def _render(self, mode='rgb_array', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        img = self.get_raw_img()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
