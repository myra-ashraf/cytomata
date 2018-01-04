# Game Settings
DEBUG = True
NO_DISPLAY = False
TITLE = "Cytomata"
WIDTH = 400
HEIGHT = 400
FPS = 30
GAME_SPEED = 1
TILESIZE = 80
GRIDWIDTH = WIDTH // TILESIZE
GRIDHEIGHT = HEIGHT // TILESIZE
MAX_TIME = 15.0
# MAP_FILE = 'm0.txt'
BKG_IMG = 'white_plaster.png'
MUSIC = 'greySector.mp3'

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BGCOLOR = (240, 240, 240)
DARKGREY = (50, 50, 50)
LIGHTGREY = (100, 100, 100)
LIGHTBLUE = (187, 222, 251)
BLUE = (25,107,189)
RED = (194, 40, 40)
GREEN = (54,143,59)
YELLOW = (253, 215, 40)

# Cyte Settings
NUM_RANDOM_CYTES = 0
CYTE_IMG = 'epi1.png'

# Proxy Settings
NUM_RANDOM_PROXIES = 1
PROXY_SPEED = 90.0
PROXY_FRICTION = 0.9
CONTROL_SCHEME = 'joystick'
PROXY_IMG = 'mac1.png'
EAT_SND = 'eat.wav'

# Cancer Settings
NUM_RANDOM_CANCERS = 1
CANCER_IMG = 'cancer2.png'

# Keymap
ACTION_MEANING = {
    0 : "NOOP",
    1 : "UP",
    2 : "DOWN",
    3 : "LEFT",
    4 : "RIGHT"}
