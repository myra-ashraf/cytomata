import sys
import os
import pygame as pg
from settings import *
from sprites import *
from tilemap import *


class Game(object):
    """Game manager"""
    def __init__(self):
        """Initialize game, window, etc."""
        pg.init()
        self.screen = pg.display.set_mode((WIDTH, HEIGHT))
        pg.display.set_caption(TITLE)
        self.clock = pg.time.Clock()
        pg.key.set_repeat(100, 100)
        self.load_data()

    def load_data(self):
        game_dir = os.path.dirname(__file__)
        self.map = Map(os.path.join(game_dir, 'maps', 'm2.txt'))

    def new(self):
        """Set up vars/objects for new game"""
        self.all_sprites = pg.sprite.Group()
        self.walls = pg.sprite.Group()
        for row, tiles in enumerate(self.map.data):
            for col, tile in enumerate(tiles):
                if tile == '1':
                    Wall(self, col, row)
                if tile == 'P':
                    self.player = Player(self, col, row)
        self.camera = Camera(self.map.width, self.map.height)

    def run(self):
        """Game loop"""
        self.playing = True
        while self.playing:
            self.dt = self.clock.tick(FPS) / 1000
            self.events()
            self.update()
            self.draw()

    def quit(self):
        """Quit to desktop"""
        pg.quit()
        sys.exit()

    def update(self):
        """Game loop - updates"""
        self.all_sprites.update()
        self.camera.update(self.player)

    def draw_grid(self):
        """Draw the tiles based on settings"""
        for x in range(0, WIDTH, TILESIZE):
            pg.draw.line(self.screen, LIGHTGREY, (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, TILESIZE):
            pg.draw.line(self.screen, LIGHTGREY, (0, y), (WIDTH, y))

    def draw(self):
        """Game loop - render"""
        self.screen.fill(LIGHTBLUE)
        self.draw_grid()
        for sprite in self.all_sprites:
            self.screen.blit(sprite.image, self.camera.apply(sprite))
        pg.display.flip()

    def events(self):
        """Game loop - process inputs/events"""
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.quit()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    self.quit()

    def show_start_screen(self):
        """Game start screen"""
        pass

    def show_go_screen(self):
        """Game over screen"""
        pass

# Executed code
g = Game()
g.show_start_screen()
while True:
    g.new()
    g.run()
    g.show_go_screen()