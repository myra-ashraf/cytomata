import sys
import os
import random as rnd
import pygame as pg
import pymunk as pm
from .settings import *
from .sprites import *
from .tilemap import *


class Game(object):
    """Game manager"""
    def __init__(self):
        """Initialize game, window, etc."""
        pg.mixer.pre_init(44100, -16, 1, 512)
        pg.init()
        self.screen = pg.display.set_mode((WIDTH, HEIGHT))
        pg.display.set_caption(TITLE)
        # Clock for setting tempo of game
        self.clock = pg.time.Clock()
        # Time before key hold detected + frequency of key press
        pg.key.set_repeat(100, 100)
        self.load_data()

    def load_data(self):
        """Load maps, sprites, and other game resources"""
        game_dir = os.path.dirname(__file__)
        img_dir = os.path.join(game_dir, 'img')
        snd_dir = os.path.join(game_dir, 'snd')
        music_dir = os.path.join(game_dir, 'music')
        self.map = Map(os.path.join(game_dir, 'maps', MAP_FILE))
        # self.bkg_img = pg.image.load(os.path.join(img_dir, 'bkgd', GRND_IMG)).convert()
        # self.bkg_rect = self.bkg_img.get_rect()
        self.proxy_img = pg.image.load(os.path.join(img_dir, 'chars', PROXY_IMG)).convert_alpha()
        self.proxy_img = pg.transform.scale(self.proxy_img, (TILESIZE, TILESIZE))
        self.cyte_img = pg.image.load(os.path.join(img_dir, 'chars', CYTE_IMG)).convert_alpha()
        self.cyte_img = pg.transform.scale(self.cyte_img, (TILESIZE, TILESIZE))
        self.cancer_img = pg.image.load(os.path.join(img_dir, 'chars', CANCER_IMG)).convert_alpha()
        self.cancer_img = pg.transform.scale(self.cancer_img, (TILESIZE, TILESIZE))
        self.eat_snd = pg.mixer.Sound(os.path.join(snd_dir, EAT_SND))
        pg.mixer.music.load(os.path.join(music_dir, MUSIC))
        pg.mixer.music.set_volume(1.0)

    def new(self):
        """Set up objects (sprites, camera, etc.) for new game"""
        self.space = pm.Space()
        self.space.gravity = (0.0, 0.0)
        self.space.add_collision_handler(0, 2).post_solve=self.proxy_cancer_collision
        self.space.add_collision_handler(0, 1).post_solve=self.proxy_cyte_collision
        self.all_sprites = []
        self.proxies = []
        self.cytes = []
        self.cancers = []
        self.score = 0
        self.last_update = pg.time.get_ticks()
        self.spawn_from_map()
        self.spawn_randomly(Cyte, NUM_RANDOM_CYTES)
        self.spawn_randomly(Cancer, NUM_RANDOM_CANCERS)
        self.spawn_randomly(Proxy, NUM_RANDOM_PROXIES)
        # self.camera = Camera(self.map.width, self.map.height)
        pg.mixer.music.play(-1)

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

    def get_open_spots(self):
        """Return grid coordinates of locations on the map not occupied by some object"""
        open_spots = [
            (x, y) for x in range(round(int(self.map.tile_width)))
            for y in range(round(int(self.map.tile_height)))
            if (x, y) not in self.map.occupied_tiles
        ]
        return open_spots

    def spawn_randomly(self, Entity, number):
        """Creates a proxy/cyte/cancer in a random spot on the map"""
        open_spots = self.get_open_spots()
        for i in range(number):
            rand_x, rand_y = rnd.choice(open_spots)
            Entity(self, rand_x, rand_y)
            open_spots.remove((rand_x, rand_y))
            self.map.occupied_tiles.append((rand_x, rand_y))

    def run(self):
        """Game loop"""
        self.playing = True
        while self.playing:
            self.dt = self.clock.tick(FPS) / 1000
            self.events()
            self.update()
            self.draw()
            self.clock.tick(FPS)


    def events(self):
        """Game loop - process inputs/events"""
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.quit()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    self.quit()

    def update(self):
        """Game loop - updates"""
        for sprite in self.all_sprites:
            sprite.update()
        self.time_penalty(3000)
        # Camera tracking
        # self.camera.update(self.proxy)
        # End the current game if all cancers have been eliminated
        if len(self.cancers) < 1:
            self.playing = False

    def draw(self):
        """Game loop - render"""
        if DEBUG:
            pg.display.set_caption('{:.2f}'.format(self.clock.get_fps()))
        self.screen.fill(BGCOLOR)
        # self.screen.blit(self.bkg_img, self.bkg_rect)
        # self.draw_grid()
        for sprite in self.all_sprites:
            self.draw_sprite(self.screen, sprite, sprite.image)
        self.draw_text(self.screen, str(self.score), 18, WIDTH/2, 10)
        self.space.step(1.0/FPS)
        pg.display.flip()

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
        proxy_body = a.body
        cancer_body = b.body
        for cancer in self.cancers:
            if cancer_body == cancer.body:
                self.score += 15
                self.space.remove(cancer.shape, cancer.shape.body)
                self.cancers.remove(cancer)
                self.all_sprites.remove(cancer)

    def proxy_cyte_collision(self, arbiter, space, _):
        """Collision between bird and pig"""
        a, b = arbiter.shapes
        proxy_body = a.body
        cyte_body = b.body
        for cyte in self.cytes:
            if cyte_body == cyte.body and not cyte.shield:
                cyte.life -= 1
                if cyte.life <= 0:
                    self.space.remove(cyte.shape, cyte.shape.body)
                    self.cytes.remove(cyte)
                    self.all_sprites.remove(cyte)
                    self.score -= 5
                else:
                    cyte.shield = 8

    def time_penalty(self, duration):
        now = pg.time.get_ticks()
        if now - self.last_update > duration:
            self.last_update = now
            self.score -= 1

    def show_start_screen(self):
        """Game start screen"""
        pass

    def show_go_screen(self):
        """Game over screen"""
        pass

    def quit(self):
        """Quit to desktop"""
        pg.quit()
        sys.exit()

# Executed code
def run():
    g = Game()
    g.show_start_screen()
    while True:
        g.new()
        g.run()
        g.show_go_screen()
