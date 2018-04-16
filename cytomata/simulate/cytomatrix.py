import os
import sys

import gym
import numpy as np
import pygame as pg
import pymunk as pm


###################
## Game Settings ##
###################
# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Main Settings
DISPLAY_SCREEN = True
SHOW_STATS = True
TITLE = "Cytomata"
WIDTH = 200
HEIGHT = 200
TILESIZE = 20
GAME_SPEED = 0.1 * TILESIZE / 40
# MAX_TIME = 10.0
TERMINAL_STEP = 8000
ACTION_MEANING = {0: "NOOP", 1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT"}

# Sprite Settings
NUM_RANDOM_PROXIES = 1
PROXY_SPEED = 45.0
PROXY_MAX_SPEED = 90.0
PROXY_FRICTION = 0.9
PROXY_IMG = 'mac2.png'
NUM_RANDOM_CYTES = 30
CYTE_IMG = 'blank.png'
NUM_RANDOM_CANCERS = 1
CANCER_IMG = 'cancer2.png'

#########################
## Game System Classes ##
#########################
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
        self.action_space = gym.spaces.Discrete(len(self._action_set))
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(WIDTH, HEIGHT, 3), dtype=np.uint8)
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


class Map(object):
    def __init__(self, filename):
        self.data = []
        with open(filename, 'rt') as f:
            for line in f:
                self.data.append(line.strip())
        self.tile_width = len(self.data[0])
        self.tile_height = len(self.data)
        self.width = self.tile_width * TILESIZE
        self.height = self.tile_height * TILESIZE


class Camera(object):
    def __init__(self, width, height):
        self.camera = pg.Rect(0, 0, width, height)
        self.width = width
        self.height = height

    def apply(self, entity):
        return entity.rect.move(self.camera.topleft)

    def update(self, target):
        x = -target.rect.centerx + int(WIDTH/2)
        y = -target.rect.centery + int(HEIGHT/2)
        # limit scrolling to map size
        x = min(0, x) # left
        y = min(0, y) # top
        x = max(-(self.width - WIDTH), x) # right
        y = max(-(self.height - HEIGHT), y) # bottom
        self.camera = pg.Rect(x, y, self.width, self.height)


####################
## Sprite Classes ##
####################
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
        self.random_walk(64, 20)

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
