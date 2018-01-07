import os
import sys

import gym
from gym import spaces
import numpy as np
import pygame as pg
from pygame.math import Vector2


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

DISPLAY_SCREEN = True
SHOW_STATS = True
TITLE = 'Pong'
WIDTH = 400
HEIGHT = 400
FPS = 60

MAX_SCORE = 21  # Max points by either agent or opp that results in terminal state
AGENT_SPEED_RATIO = 1.5  # Speed of player contingent on HEIGHT
OPP_SPEED_RATIO = 1.4  # Speed of opponent contingent on HEIGHT
BALL_SPEED_RATIO = 1.4  # Speed of ball contingent on HEIGHT
PADDLE_WIDTH_RATIO = 0.05  # Paddle width contingent on WIDTH
PADDLE_HEIGHT_RATIO = 0.1  # Paddle height contingent on HEIGHT
PADDLE_SPACING_RATIO = 0.04  # Width of space btw paddle & wall contingent on WIDTH
BALL_RADIUS_RATIO = 0.02  # Width of ball contingent on HEIGHT
REWARD_MINOR = 0.1  # Reward for hitting the ball
REWARD_MAJOR = 1.0  # Reward for getting ball past opponent
ACTION_MEANING = {0: "NOOP", 1: "UP", 2: "DOWN"}  # External input to game action


class PygamePong(gym.Env):
    """
    Based on Chris Bradfield's pygame project structure:
    https://github.com/kidscancode/pygame_tutorials
    and modified from PLE's pong:
    https://github.com/ntasfi/PyGame-Learning-Environment
    which is derived from marti1125's pong game:
    https://github.com/marti1125/pong/
    """
    def __init__(self):
        if not DISPLAY_SCREEN:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        pg.init()
        self.screen = pg.display.set_mode((WIDTH, HEIGHT), 0, 32)
        pg.display.set_caption(TITLE)
        self.clock = pg.time.Clock()
        self.episode_framesteps = []
        self._action_set = range(0, len(ACTION_MEANING))
        self.action_space = spaces.Discrete(len(self._action_set))
        self.observation_space = spaces.Box(low=0, high=255, shape=(WIDTH, HEIGHT, 3))
        self.reward_range = (-np.inf, np.inf)

    def _reset(self):
        self.new()
        self.draw()
        return self.get_raw_img()

    def new(self):
        self.terminal = False
        self.scoreboard = {'agent': 0, 'opp': 0}
        self.reward = 0.0
        self.framestep = 0
        self.spawn_sprites()

    def spawn_sprites(self):
        self.all_sprites = pg.sprite.Group()
        self.paddles = pg.sprite.Group()
        self.agent = Paddle(self, 'agent')
        self.opp = Paddle(self, 'opp')
        self.ball = Ball(self)

    def _step(self, action):
        self.reward = 0.0
        self.framestep += 1
        self.dt = self.clock.tick(FPS) / 1000.0
        self.events()
        self.update(action)
        self.draw()
        if self.terminal:
            self.episode_framesteps.append(self.framestep)
        return self.get_raw_img(), self.reward, self.terminal, self.get_nonvis_state()

    def events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT or (
                event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                pg.quit()
                sys.exit()

    def update(self, action):
        for sprite in self.all_sprites:
            sprite.update(action)
        self.check_scored()
        self.terminal = max(list(self.scoreboard.values())) >= MAX_SCORE

    def check_scored(self):
        if self.ball.pos.x > WIDTH:
            self.reward += REWARD_MAJOR
            self.scoreboard['agent'] += 1
            self.spawn_sprites()
        elif self.ball.pos.x < 0:
            self.reward -= REWARD_MAJOR
            self.scoreboard['opp'] += 1
            self.spawn_sprites()

    def draw(self):
        if not DISPLAY_SCREEN:
            self.screen = pg.Surface((WIDTH, HEIGHT), pg.SRCALPHA, 32)
        self.screen.fill(BLACK)
        for sprite in self.all_sprites:
            self.screen.blit(sprite.image, sprite.rect.topleft)
        if SHOW_STATS:
            self.show_stats()
        pg.display.update()

    def show_stats(self):
        fps_str = 'FPS: {:.2f}'.format(self.clock.get_fps())
        score_str = ' | Score: {} - {}'.format(
            self.scoreboard['agent'], self.scoreboard['opp']
        )
        pg.display.set_caption(fps_str + score_str)

    def get_raw_img(self):
        screen_surface = pg.transform.flip(
            pg.transform.rotate(self.screen, 90), False, True
        )
        return pg.surfarray.array3d(screen_surface)

    def get_nonvis_state(self):
        # For training on non-visual info instead of raw pixels
        return {
            'framestep': self.framestep,
            'episode_framesteps': self.episode_framesteps,
            'player_y': self.agent.pos.y,
            'player_velocity': self.agent.vel.y,
            'opp_y': self.opp.pos.y,
            'ball_x': self.ball.pos.x,
            'ball_y': self.ball.pos.y,
            'ball_velocity_x': self.ball.vel.x,
            'ball_velocity_y': self.ball.vel.y
        }

    def get_action_meanings(self):
        return [ACTION_MEANING[i] for i in self._action_set]


class Paddle(pg.sprite.Sprite):

    def __init__(self, game, ptype, posit_y=HEIGHT/2):
        pg.sprite.Sprite.__init__(self)
        self.game = game
        self.ptype = ptype
        if ptype == 'agent':
            self.speed = HEIGHT * AGENT_SPEED_RATIO
            self.pos = Vector2(
                int(np.round(WIDTH * PADDLE_SPACING_RATIO)), posit_y
            )
        elif ptype == 'opp':
            self.speed = HEIGHT * OPP_SPEED_RATIO
            self.pos = Vector2(
                WIDTH - int(np.round(WIDTH * PADDLE_SPACING_RATIO)), posit_y
            )
        self.vel = Vector2(0.0, 0.0)
        self.rect_width = int(np.round(WIDTH * PADDLE_WIDTH_RATIO))
        self.rect_height = int(np.round(HEIGHT * PADDLE_HEIGHT_RATIO))
        self.image = pg.Surface((self.rect_width, self.rect_height))
        self.image.fill(WHITE)
        self.image.set_colorkey((0, 0, 0))
        self.rect = self.image.get_rect()
        self.rect.center = self.pos
        self.game.all_sprites.add(self)
        self.game.paddles.add(self)

    def update(self, action):
        self.vel.y = 0.0
        if self.ptype == 'agent':
            self.check_inputs()
            if action is not None:
                self.move(ACTION_MEANING[action])
        if self.ptype == 'opp':
            self.scripted_response()
        self.resolve_collisions()
        self.rect.centery = self.pos.y

    def check_inputs(self):
        keys = pg.key.get_pressed()
        if keys[pg.K_UP] or keys[pg.K_w]:
            self.move('UP')
        if keys[pg.K_DOWN] or keys[pg.K_s]:
            self.move('DOWN')

    def move(self, direction):
        if direction == 'UP':
            self.vel.y = -self.speed * self.game.dt
        elif direction == 'DOWN':
            self.vel.y = self.speed * self.game.dt
        self.pos.y += self.vel.y
        self.rect.centery = self.pos.y

    def resolve_collisions(self):
        # Collide with top of screen
        self.rect.centery = self.pos.y
        if self.pos.y - self.rect_height / 2 <= 0:
            self.pos.y = self.rect_height / 2
            self.vel.y = 0.0
        # Collide with bottom of screen
        if self.pos.y + self.rect_height / 2 >= HEIGHT:
            self.pos.y = HEIGHT - self.rect_height / 2
            self.vel.y = 0.0
        self.rect.centery = self.pos.y

    def scripted_response(self):
        incoming = (
            self.game.ball.vel.x >= 0 and
            self.game.ball.pos.x >= WIDTH / 2
        )
        if self.ptype == 'agent':
            incoming = not incoming
        if incoming:
            # When ball heading towards this paddle
            if self.pos.y < self.game.ball.pos.y - self.rect_height / 8.0:
                self.vel.y = self.speed / 1.5 * self.game.dt
            elif self.pos.y > self.game.ball.pos.y + self.rect_height / 8.0:
                self.vel.y = -self.speed / 1.5 * self.game.dt
        else:
            # When ball heading away from this paddle
            if self.pos.y < HEIGHT / 4.0 - self.rect_height / 8.0:
                self.vel.y = self.speed / 6.0 * self.game.dt
            if self.pos.y > HEIGHT / 4.0 + self.rect_height / 8.0:
                self.vel.y = -self.speed / 6.0 * self.game.dt
        self.pos.y += self.vel.y
        self.rect.centery = self.pos.y


class Ball(pg.sprite.Sprite):

    def __init__(self, game):
        pg.sprite.Sprite.__init__(self)
        self.game = game
        self.radius = int(np.round(HEIGHT * BALL_RADIUS_RATIO))
        self.speed = HEIGHT * BALL_SPEED_RATIO
        self.pos = Vector2(WIDTH / 2, HEIGHT / 2)
        # vel_x = int(np.round(np.random.choice([-1.0, 1.0]) * self.speed))
        vel_x = int(np.round(self.speed))
        vel_y = int(np.round(np.random.uniform(-0.2, 0.2) * self.speed))
        self.vel = Vector2(vel_x, vel_y)
        self.image = pg.Surface((self.radius * 2, self.radius * 2))
        self.image.fill((0, 0, 0, 0))
        self.image.set_colorkey((0, 0, 0))
        pg.draw.circle(self.image, WHITE, (self.radius, self.radius), self.radius, 0)
        self.rect = self.image.get_rect()
        self.rect.center = self.pos
        self.game.all_sprites.add(self)

    def update(self, action):
        self.pos.x += self.vel.x * self.game.dt
        self.pos.y += self.vel.y * self.game.dt
        self.rect.center = self.pos
        self.resolve_collisions()
        self.rect.center = self.pos

    def resolve_collisions(self):
        # Hits a paddle
        hits = pg.sprite.spritecollide(
            self, self.game.paddles, False, pg.sprite.collide_rect_ratio(1.02)
        )
        if hits:
            if hits[0] == self.game.agent:
                self.game.reward += REWARD_MINOR
            if self.vel.x < 0:
                self.pos.x = hits[0].rect.right + self.radius
                self.vel.x = -self.vel.x + self.speed * 0.05
            elif self.vel.x >= 0:
                self.pos.x = hits[0].rect.left - self.radius
                self.vel.x = -self.vel.x - self.speed * 0.05
            self.vel.y += hits[0].vel.y * 8.0
            self.pos += self.vel * self.game.dt
            self.rect.center = self.pos
        # Hits the top or bottom of screen
        if (self.pos.y - self.radius <= 0) or (self.pos.y + self.radius >= HEIGHT):
            self.vel.y = -self.vel.y
            self.pos.y += self.vel.y * self.game.dt
            self.rect.center = self.pos
