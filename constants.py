import os
import pygame as pg

vec = pg.math.Vector2

# Main window 
TITLE = "Vol3mon"
WIDTH = 500
HEIGHT = 500
FPS = 60

# Directories
FILE_DIR = os.path.dirname(__file__)
IMAGES_DIR = os.path.join(FILE_DIR, "../imgs")
SNAP_FOLDER = os.path.join(FILE_DIR, "../snapshots")

# Colors
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)
GREEN2 = (39, 151, 0)
GREEN3 = (102, 203, 112)
ORANGE = (255, 127, 0)
BACKGROUND = (30, 30, 30)

# Player
PLAYER_X_SPEED = 3
PLAYER_Y_SPEED = 3




