import pygame.midi
import pygame
import time

pygame.midi.init()

print(pygame.midi.get_default_input_id())
print(pygame.midi.get_device_info(0))

player = pygame.midi.Output(0)
player.set_instrument(48, 1)

major = [0, 4, 7, 12]

def go(note):
    player.note_on(note, 127, 1)
    time.sleep(1)
    player.note_off(note, 127, 1)

def arp(base, ints):
    for n in ints:
        go(base+n)

def chord(base, ints):
    player.note_on(base+ints[0], 127, 1)
    player.note_on(base+ints[1], 127, 1)
    player.note_on(base+ints[2], 127, 1)
    player.note_on(base+ints[3], 127, 1)
    time.sleep(1)
    player.note_on(base + ints[0], 127, 1)
    player.note_on(base + ints[1], 127, 1)
    player.note_on(base + ints[2], 127, 1)
    player.note_on(base + ints[3], 127, 1)

def end():
    pygame.midi.quit()

if __name__ == '__main__':
    chord(60, major)
    end()
