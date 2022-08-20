"""
Sprites classes
"""

import math
import pygame as pg
import random
import numpy as np

from math import (cos, degrees, sin, tan, acos, 
                  atan, atan2, pi, radians, sqrt)
from itertools import cycle
import constants as const
vec = pg.math.Vector2

class Obstacle(pg.sprite.Sprite):
    def __init__(self, game, x, y, w, h, color=None, vel=vec(0, 0)):
        pg.sprite.Sprite.__init__(self)
        self.game = game
        self.pos = vec(x, y)
        self.image = pg.Surface((w, h))
        if color is not None:
            self.image.fill(color)
        self.vel = vel
        self.rect = self.image.get_rect()
        self.rect.center = self.pos
        self.old_rect = self.rect.copy()
        
    def update(self) -> None:
        if self.vel != vec(0, 0):
            # Old frame rect
            self.old_rect = self.rect.copy()
            # Updating position
            self.pos.y += self.vel.y
            self.rect.center = self.pos
            # Collisions
            self.collisions()

    def collisions(self) -> None:
        # Ceiling collision
        if self.rect.top < 0:
            self.rect.top = 0
            self.pos.y = self.rect.centery
            self.vel.y *= -1
        # Net collision
        if self.rect.colliderect(self.game.net):
            self.rect.bottom = self.game.net.rect.top
            self.pos.y = self.rect.centery
            self.vel.y *= -1
        # Immobile ball collision (those two cases rarely happen)
        if self.rect.colliderect(self.game.player):
            if self.game.player.is_standing(1, False, [self.game.net]):
                self.rect.bottom = self.game.player.rect.top
                self.pos.y = self.rect.centery
                self.vel.y *= -1
            if self.game.player.rect.top < 0:
                self.rect.top = self.game.player.rect.bottom
                self.pos.y = self.rect.centery
                self.vel.y *= -1

def main():
    pass

if __name__ == "__main__":
    main()




