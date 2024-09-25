import os
import random
import curses

from utlities import Terminal
from time import sleep
from subprocess import call

M_TOP_RIGHT = """ /$$      /$$
| $$$    /$$$
| $$$$  /$$$$
| $$ $$/$$ $$
| $$  $$$| $$
| $$\\  $ | $$
| $$ \\/  | $$
|__/     |__/"""

M_TOP_LEFT = """$$\\      $$\\
$$$\\    $$$ |
$$$$\\  $$$$ |
$$\\$$\\$$ $$ |
$$ \\$$$  $$ |
$$ |\\$  /$$ |
$$ | \\_/ $$ |
\\__|     \\__|"""

M_BOTTOM_LEFT = """ __       __ 
/  \\     /  |
$$  \\   /$$ |
$$$  \\ /$$$ |
$$$$  /$$$$ |
$$ $$ $$/$$ |
$$ |$$$/ $$ |
$$ | $/  $$ |
$$/      $$/ """

M_BOTTOM_RIGHT = """ __       __ 
|  \\     /  \\
| $$\\   /  $$
| $$$\\ /  $$$
| $$$$\\  $$$$
| $$\\$$ $$ $$
| $$ \\$$$| $$
| $$  \\$ | $$
 \\$$      \\$$"""


class WordBlock:
    @staticmethod
    def prepare_word_block(word_block: str):
        max_row = 0
        max_col = 0;

        lines = word_block.split("\n")

        for line in lines:
            if len(line) > max_col:
                max_col = len(line)
            max_row += 1

        for i in range(len(lines)):
            lines[i] = lines[i].ljust(max_col)

        return "\n".join(lines)

    def __init__(self, word_block: str):
        self.word_block = self.prepare_word_block(word_block)
        self.lines = word_block.split("\n")
        self.width = len(self.lines[0])
        self.height = len(self.lines)


class App:
    @staticmethod
    def clear_screen():
        call(["clear"])

    def __init__(self):
        self.x = 0
        self.y = 0

        self.direction_x = 1
        self.direction_y = 1

        self.screen_width, self.screen_height = os.get_terminal_size()

        self.top_left = WordBlock(M_TOP_LEFT)
        self.top_right = WordBlock(M_TOP_RIGHT)
        self.bottom_left = WordBlock(M_BOTTOM_LEFT)
        self.bottom_right = WordBlock(M_BOTTOM_RIGHT)
        self.current_word_block = self.bottom_right

        self.current_color = f"\033[38;5;{random.randint(0, 255)}m"

    def run(self):
        try:
            while True:
                self.clear_screen()
                self.update()
                sleep(0.2)
        except KeyboardInterrupt:
            self.clear_screen()
            print("Goodbye!")

    def update(self):
        self.screen_width, self.screen_height = os.get_terminal_size()

        for _ in range(self.y):
            print()

        for line in self.current_word_block.lines:
            print(self.current_color + (self.x * " ") + line + Terminal.END)

        self.x += self.direction_x
        self.y += self.direction_y

        if self.x <= 0 or self.x + self.current_word_block.width >= self.screen_width:
            self.direction_x *= -1
            self.switch_word_block()
        
        if self.y <= 0 or self.y + self.current_word_block.height + 1 >= self.screen_height:
            self.direction_y *= -1
            self.switch_word_block()

    def switch_word_block(self):
        if self.direction_x == 1 and self.direction_y == 1:
            self.current_word_block = self.bottom_right
        elif self.direction_x == 1 and self.direction_y == -1:
            self.current_word_block = self.top_right
        elif self.direction_x == -1 and self.direction_y == 1:
            self.current_word_block = self.bottom_left
        elif self.direction_x == -1 and self.direction_y == -1:
            self.current_word_block = self.top_left

        if self.x + self.current_word_block.width > self.screen_width:
            self.x = self.screen_width - self.current_word_block.width
        
        if self.y + self.current_word_block.height + 1 > self.screen_height:
            self.y = self.screen_height - self.current_word_block.height - 1
        
        self.current_color = f"\033[38;5;{random.randint(0, 255)}m"


class CursesApp:
    instance = None

    def __init__(self) -> None:
        CursesApp.instance = self

        self.top_left = WordBlock(M_TOP_LEFT)
        self.top_right = WordBlock(M_TOP_RIGHT)
        self.bottom_left = WordBlock(M_BOTTOM_LEFT)
        self.bottom_right = WordBlock(M_BOTTOM_RIGHT)
        self.current_word_block = self.bottom_right

        self.x = 0
        self.y = 0

        self.direction_x = 1
        self.direction_y = 1

        self.current_color = None

        self.stdsrc = None
    
    def draw_word_block(self, word_block: WordBlock, x_offset: int, y_offset: int):
        x = x_offset
        y = y_offset
        for line in word_block.lines:
            if x >= curses.COLS:
                break
            if x + word_block.width > curses.COLS:
                line = line[:curses.COLS - x]
            if y > curses.LINES:
                break

            self.stdsrc.addstr(y, x, line, self.current_color)
            y += 1

    def switch_word_block(self):
        if self.direction_x == 1 and self.direction_y == 1:
            self.current_word_block = self.bottom_right
        elif self.direction_x == 1 and self.direction_y == -1:
            self.current_word_block = self.top_right
        elif self.direction_x == -1 and self.direction_y == 1:
            self.current_word_block = self.bottom_left
        elif self.direction_x == -1 and self.direction_y == -1:
            self.current_word_block = self.top_left

        # if self.x + self.current_word_block.width > self.screen_width:
        #     self.x = self.screen_width - self.current_word_block.width
        
        # if self.y + self.current_word_block.height + 1 > self.screen_height:
        #     self.y = self.screen_height - self.current_word_block.height - 1
        
        self.current_color = curses.color_pair(random.randint(1, 255))

    @staticmethod
    def run(stdsrc):
        self = CursesApp.instance
        self.stdsrc = stdsrc

        curses.start_color()
        curses.use_default_colors()
        for i in range(0, curses.COLORS):
            try:
                curses.init_pair(i + 1, i, -1)
            except:
                pass
        
        self.current_color = curses.color_pair(100)

        stdsrc.clear()
        curses.curs_set(0)

        while True:
            try:
                sleep(0.1)
            except KeyboardInterrupt:
                break
            # key = stdsrc.getch()

            stdsrc.clear()

            self.x += self.direction_x
            self.y += self.direction_y

            if self.x <= 0 or (self.x + self.current_word_block.width >= curses.COLS):
                self.direction_x *= -1
                self.x += self.direction_x * 2
                self.switch_word_block()
            
            if self.y < 0 or (self.y + self.current_word_block.height >= curses.LINES):
                self.direction_y *= -1
                self.y += self.direction_y * 2
                self.switch_word_block()

            self.draw_word_block(self.current_word_block, self.x, self.y)

            stdsrc.refresh()


if __name__ == "__main__":
    # app = App()
    # app.run()

    app = CursesApp()
    curses.wrapper(CursesApp.run)
