import pygame as pg
import math
import matplotlib.pyplot as plt
from IPython import display

plt.ion()
pg.init()

def plot(scores, mean_scores, file_name):
    display.clear_output(wait=True)
    plt.clf()
    plt.title("training...")
    plt.xlabel("n games")
    plt.ylabel("score")
    plt.plot(scores, "b")
    plt.plot(mean_scores, "r")
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.savefig(file_name, bbox_inches="tight", pad_inches=0.025)
    plt.show(block=False)

def message(screen, msg, font_size, color, position) -> None:
    """
    Displays a message on screen.

    Parameters
    ----------
    msg: string (required)
        Actual text to be displayed
    font_size: int (required)
        Font size
    color: tuple of int (required)
        Text RGB color code
    position: tuple of int (required)
        Position on screen
    """ 
    font = pg.font.SysFont("Calibri", font_size)
    text = font.render(msg, True, color)
    text_rect = text.get_rect(topleft=(position))
    screen.blit(text, text_rect)

def distance(p1, p2) -> float:
    """Returns the distance between two points p1, p2."""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)