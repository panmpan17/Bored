import os

from time import sleep
from subprocess import call

M_1 = """ /$$      /$$
| $$$    /$$$
| $$$$  /$$$$
| $$ $$/$$ $$
| $$  $$$| $$
| $$\\  $ | $$
| $$ \\/  | $$
|__/     |__/"""

M_2 = """$$\\      $$\\
$$$\\    $$$ |
$$$$\\  $$$$ |
$$\\$$\\$$ $$ |
$$ \\$$$  $$ |
$$ |\\$  /$$ |
$$ | \\_/ $$ |
\\__|     \\__|"""

M_3 = """ __       __ 
/  \\     /  |
$$  \\   /$$ |
$$$  \\ /$$$ |
$$$$  /$$$$ |
$$ $$ $$/$$ |
$$ |$$$/ $$ |
$$ | $/  $$ |
$$/      $$/ """

M_4 = """ __       __ 
|  \\     /  \\
| $$\\   /  $$
| $$$\\ /  $$$
| $$$$\\  $$$$
| $$\\$$ $$ $$
| $$ \\$$$| $$
| $$  \\$ | $$
 \\$$      \\$$"""


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


class WordBlock:
    def __init__(self, word_block: str):
        self.word_block = prepare_word_block(word_block)
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

        self.worl_blocks = [WordBlock(M_1), WordBlock(M_2), WordBlock(M_3), WordBlock(M_4)]
        self.current_word_block = self.worl_blocks[0]
        self.worl_block_index = 0

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
            print((self.x * " ") + line)

        self.x += self.direction_x
        self.y += self.direction_y

        if self.x <= 0 or self.x + self.current_word_block.width >= self.screen_width:
            self.direction_x *= -1
            self.switch_word_block()
        
        if self.y <= 0 or self.y + self.current_word_block.height + 1 >= self.screen_height:
            self.direction_y *= -1
            self.switch_word_block()
    
    def switch_word_block(self):
        self.worl_block_index = (self.worl_block_index + 1) % len(self.worl_blocks)
        self.current_word_block = self.worl_blocks[self.worl_block_index]

        if self.x + self.current_word_block.width > self.screen_width:
            self.x = self.screen_width - self.current_word_block.width
        
        if self.y + self.current_word_block.height + 1 > self.screen_height:
            self.y = self.screen_height - self.current_word_block.height - 1




# M_1 = prepare_word_block(M_1)
# print(prepare_word_block(M_2))
# print(prepare_word_block(M_3))
# print(prepare_word_block(M_4))

if __name__ == "__main__":
    app = App()
    app.run()
