import sys
import os
import random as rnd
import pygame as pg
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
        self.cancer_img = pg.image.load(os.path.join(img_dir, 'chars', CANCER_IMG)).convert_alpha()
        self.cancer_img = pg.transform.scale(self.cancer_img, (TILESIZE, TILESIZE))
        self.cell_img = pg.image.load(os.path.join(img_dir, 'bkgd', CELL_IMG)).convert_alpha()
        self.cell_img = pg.transform.scale(self.cell_img, (TILESIZE, TILESIZE))
        self.eat_snd = pg.mixer.Sound(os.path.join(snd_dir, EAT_SND))
        pg.mixer.music.load(os.path.join(music_dir, MUSIC))
        pg.mixer.music.set_volume(1.0)

    def new(self):
        """Set up objects (sprites, camera, etc.) for new game"""
        self.all_sprites = pg.sprite.Group()
        self.proxies = pg.sprite.Group()
        self.cells = pg.sprite.Group()
        self.cancers = pg.sprite.Group()
        self.score = 0
        for row, tiles in enumerate(self.map.data):
            for col, tile in enumerate(tiles):
                if tile == '0':
                    self.map.occupied_tiles.append((col, row))
                    Cell(self, col, row)
                if tile == '1':
                    self.map.occupied_tiles.append((col, row))
                    Proxy(self, col, row)
                if tile == '2':
                    self.map.occupied_tiles.append((col, row))
                    Cancer(self, col, row)
        if SPAWN_CELLS_RANDOMLY:
            allowed = [
                (x, y) for x in range(round(int(self.map.width/TILESIZE)))
                for y in range(round(int(self.map.height/TILESIZE)))
                if (x, y) not in self.map.occupied_tiles
            ]
            for i in range(NUM_RANDOM_CELLS):
                rand_x, rand_y = rnd.choice(allowed)
                Cell(self, rand_x, rand_y)
                self.map.occupied_tiles.append((rand_x, rand_y))
        if SPAWN_CANCERS_RANDOMLY:
            allowed = [
                (x, y) for x in range(round(int(self.map.width/TILESIZE)))
                for y in range(round(int(self.map.height/TILESIZE)))
                if (x, y) not in self.map.occupied_tiles
            ]
            for i in range(NUM_RANDOM_CANCERS):
                rand_x, rand_y = rnd.choice(allowed)
                Cancer(self, rand_x, rand_y)
                self.map.occupied_tiles.append((rand_x, rand_y))
        self.camera = Camera(self.map.width, self.map.height)
        pg.mixer.music.play(-1)

    def run(self):
        """Game loop"""
        self.playing = True
        while self.playing:
            self.dt = self.clock.tick(FPS) / 1000
            self.events()
            self.update()
            self.draw()

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
        self.all_sprites.update()
        # Camera tracking
        self.camera.update(self.proxy)
        if len(self.cancers) < 1:
            self.playing = False

    def draw(self):
        """Game loop - render"""
        if DEBUG:
            pg.display.set_caption('{:.2f}'.format(self.clock.get_fps()))
        self.screen.fill(BGCOLOR)
        # self.screen.blit(self.bkg_img, self.bkg_rect)
        # self.draw_grid()
        # pg.draw.rect(self.screen, BLACK, self.camera.apply(self.proxy), 2)
        for sprite in self.all_sprites:
            self.screen.blit(sprite.image, self.camera.apply(sprite))
        self.draw_text(self.screen, str(self.score), 18, WIDTH/2, 10)
        pg.display.flip()

    def draw_grid(self):
        """Draw the tiles based on settings"""
        for x in range(0, WIDTH, TILESIZE):
            pg.draw.line(self.screen, LIGHTGREY, (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, TILESIZE):
            pg.draw.line(self.screen, LIGHTGREY, (0, y), (WIDTH, y))

    def draw_text(self, surf, text, size, x, y):
        font_name = pg.font.match_font('arial')
        font = pg.font.Font(font_name, size)
        text_surface = font.render(text, True, DARKGREY)
        text_rect = text_surface.get_rect()
        text_rect.midtop = (x, y)
        surf.blit(text_surface, text_rect)

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
