from tkinter import *
import numpy as np
import random


# Implemented by ourselves, but according to the structure of https://github.com/yangshun/2048-python


ENABLE_KEYBOARD_CONTROLS = False

LABEL_HEIGHT = 2
PADDING = 5

BG_GAME = "#92877d"
BG_EMPTY = "#9e948a"
BG_TILE = {2: "#eee4da", 4: "#ede0c8", 8: "#f2b179", 16: "#f59563", 32: "#f67c5f", 64: "#f65e3b", 128: "#edcf72",
           256: "#edcc61", 512: "#edc850", 1024: "#edc53f", 2048: "#edc22e"}
CELL_COLOR_DICT = {2: "#776e65", 4: "#776e65", 8: "#f9f6f2", 16: "#f9f6f2", 32: "#f9f6f2", 64: "#f9f6f2",
                   128: "#f9f6f2", 256: "#f9f6f2", 512: "#f9f6f2", 1024: "#f9f6f2", 2048: "#f9f6f2"}


class Game(Frame):
    def __init__(self):
        Frame.__init__(self, bg=BG_GAME)

        self.score = 0
        self.game_state = np.zeros((4, 4), dtype=np.int16)
        self.add_tile()
        self.add_tile()

        self.grid()
        self.master.title('2048')

        if ENABLE_KEYBOARD_CONTROLS:
            self.master.bind('<Key>', self.move)

        self.score_frame = Frame(self, bg=BG_GAME)
        self.score_frame.grid()
        self.score_label = Label(master=self.score_frame, bg=BG_GAME, text=str(self.score), justify=CENTER, font=('Helvetica', 40, 'bold'))
        self.score_label.grid()

        self.tiles = []
        self.create_grid()
        self.update_grid()

        #self.mainloop()

    def restart(self):
        self.score = 0
        self.game_state = np.zeros((4, 4), dtype=np.int16)
        self.add_tile()
        self.add_tile()
        self.update_grid()

    def create_grid(self):
        board = Frame(self, bg=BG_GAME)
        board.grid()

        for row in range(4):
            tile_row = []
            for col in range(4):
                tile = Frame(board, bg=BG_EMPTY)
                tile.grid(row=row, column=col, padx=PADDING, pady=PADDING)
                label = Label(master=tile, text='', bg=BG_EMPTY, justify=CENTER, width=LABEL_HEIGHT*2,
                              height=LABEL_HEIGHT, font=('Helvetica', 40, 'bold'))
                label.grid()
                tile_row.append(label)

            self.tiles.append(tile_row)

    def update_grid(self):
        for row in range(4):
            for col in range(4):
                tile_value = self.game_state[row][col]

                if tile_value == 0:
                    self.tiles[row][col].configure(text='', bg=BG_EMPTY)
                else:
                    self.tiles[row][col].configure(text=str(tile_value), bg=BG_TILE[tile_value], fg='#776e65')

        self.score_label.configure(text=str(self.score))
        self.update()

    def add_tile(self):
        rows, cols = np.where(self.game_state == 0)
        random_number = random.randrange(0, len(rows))
        self.game_state[rows[random_number], cols[random_number]] = np.random.choice([2, 4], p=(0.8, 0.2))

    def check_termination(self):
        score_backup = self.score
        if len(np.where(self.game_state == 0)[0]) == 0:
            backup = np.copy(self.game_state)


            if self.move_up() or self.move_right() or self.move_down() or self.move_left():
                self.game_state = np.copy(backup)
                self.score = score_backup
                return False
            else:
                self.score = score_backup
                return True

    def is_valid(self, event):
        backup = np.copy(self.game_state)
        score_backup = self.score

        if self.move(event)[0]:
            self.game_state = np.copy(backup)
            self.score = score_backup
            return True

        self.score = score_backup
        return False

    def move(self, event):
        modified = False
        terminated = False

        if event.keysym == 'Up':
            modified = self.move_up()
        elif event.keysym == 'Right':
            modified = self.move_right()
        elif event.keysym == 'Down':
            modified = self.move_down()
        elif event.keysym == 'Left':
            modified = self.move_left()

        if modified:
            self.add_tile()
            self.update_grid()

            if self.check_termination():
                terminated = True

        return (modified, terminated)

    def move_left(self):
        modified = False
        for row in range(4):
            mem_col = -1

            for col in range(4):
                if self.game_state[row][col] == 0:
                    continue

                if mem_col == -1:
                    mem_col = col
                    continue

                if self.game_state[row][col] != self.game_state[row][mem_col]:
                    mem_col = col
                    continue

                if self.game_state[row][col] == self.game_state[row][mem_col]:
                    self.game_state[row][mem_col] += self.game_state[row][col]
                    self.score += int(self.game_state[row][mem_col])
                    self.game_state[row][col] = 0
                    mem_col = -1
                    modified = True

            for i in range(4*4):
                col = i % 4

                if col == 3:
                    continue

                if self.game_state[row][col] == 0 and self.game_state[row][col + 1] != 0:
                    self.game_state[row][col] = self.game_state[row][col + 1]
                    self.game_state[row][col + 1] = 0
                    modified = True

        return modified

    def move_right(self):
        modified = False
        for row in range(4):
            mem_col = -1

            for col in reversed(range(4)):
                if self.game_state[row][col] == 0:
                    continue

                if mem_col == -1:
                    mem_col = col
                    continue

                if self.game_state[row][col] != self.game_state[row][mem_col]:
                    mem_col = col
                    continue

                if self.game_state[row][col] == self.game_state[row][mem_col]:
                    self.game_state[row][mem_col] += self.game_state[row][col]
                    self.score += self.game_state[row][mem_col]
                    self.game_state[row][col] = 0
                    mem_col = -1
                    modified = True

            for i in range(4 * 4):
                col = i % 4

                if col == 0:
                    continue

                if self.game_state[row][col] == 0 and self.game_state[row][col - 1] != 0:
                    self.game_state[row][col] = self.game_state[row][col - 1]
                    self.game_state[row][col - 1] = 0
                    modified = True

        return modified

    def move_up(self):
        self.game_state = np.rot90(self.game_state)
        modified = self.move_left()
        self.game_state = np.rot90(self.game_state, 3)

        return modified

    def move_down(self):
        self.game_state = np.rot90(self.game_state)
        modified = self.move_right()
        self.game_state = np.rot90(self.game_state, 3)

        return modified
