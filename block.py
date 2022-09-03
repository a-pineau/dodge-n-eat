import pygame as pg

vec = pg.math.Vector2


class Block(pg.sprite.Sprite):
    def __init__(self, x, y, w, h, color):
        pg.sprite.Sprite.__init__(self)

        self.pos = vec(x, y)
        self.color = color
        self.image = pg.Surface((w, h))
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.rect.center = self.pos

    def draw(self, screen):
        pg.draw.rect(screen, self.color, self.rect)
