import pygame as pg
import random as rnd


# Settings
########################
TITLE = "Cytomata"
WIDTH = 360
HEIGHT = 480
FPS = 30
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (25,107,189)
RED = (194, 40, 40)
GREEN = (54,143,59)
########################


class Game():
    """Game manager"""
    def __init__(self):
        # initialize game window, etc.
        pg.init()
        pg.mixer.init()
        self.screen = pg.display.set_mode((WIDTH, HEIGHT))
        pg.display.set_caption(TITLE)
        self.clock = pg.time.Clock()
        self.running = True

    def new(self):
        # start a new game
        self.all_sprites = pg.sprite.Group()
        self.run()

    def run(self):
        # game loop
        self.playing = True
        while self.playing:
            self.clock.tick(FPS)
            self.events()
            self.update()
            self.draw()

    def events(self):
        # game loop - processing
        for event in pg.event.get():
            if event.type == pg.QUIT:
                if self.playing:
                    self.playing = False
                self.running = False

    def update(self):
        # game loop - updates
        self.all_sprites.update()

    def draw(self):
        # game loop -rendering
        self.screen.fill(BLACK)
        self.all_sprites.draw(self.screen)
        pg.display.flip()

    def show_start_screen(self):
        # game start screen
        pass

    def show_gover_screen(self):
        # game over screen
        pass

g = Game()
g.show_start_screen()
while g.running:
    g.new()
    g.show_gover_screen()

pg.quit()