import time
import os
import sys


class ProgressBar:
    @staticmethod
    def terminal_size():
        rows, columns = os.popen('stty size', 'r').read().split()
        return int(rows), int(columns)

    def __init__(self, progress_length,
                 prefix="", suffix="", decimals=1,
                 length=50, fill='█'):
        """
        Call in a loop to create terminal progress bar
        @params:
            progress_length       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
        """

        self.progress = 0
        self.progress_length = progress_length
        self.prefix = prefix
        lines = self.prefix.split('\n')
        self.prefix_line_count = len(lines) - 1
        self.suffix = suffix
        self.decimals = decimals

        percent = ("{0:." + str(self.decimals) + "f}").format(100)
        self.max_length = self.terminal_size()[1] - len(lines[-1]) - len(self.suffix) - len(percent) - 3
        self.length = length

        self.fill = fill
    
    def set_prefix(self, prefix):
        self.prefix = prefix
        lines = self.prefix.split('\n')
        self.prefix_line_count = len(lines) - 1

        percent = ("{0:." + str(self.decimals) + "f}").format(100)
        self.max_length = self.terminal_size()[1] - len(lines[-1]) - len(self.suffix) - len(percent) - 3

    def increment(self, print_new_line_if_done=False):
        self.progress += 1
        self.print(print_new_line_if_done=print_new_line_if_done)

    def set_progress(self, progress, print_new_line_if_done=False):
        self.progress = progress
        self.print(print_new_line_if_done=print_new_line_if_done)
    
    def print(self, print_new_line_if_done=False):
        length = min(self.max_length, self.length)
        percent = ("{0:." + str(self.decimals) + "f}").format(100 * (self.progress / self.progress_length))
        filled_length = int(length * self.progress // self.progress_length)
        bar = self.fill * filled_length + '-' * (length - filled_length)
        print(f'\r{self.prefix}{bar}| {percent}% {self.suffix}')
        sys.stdout.write("\x1b[1A" * (1 + self.prefix_line_count))

        # Print New Line on Complete
        if print_new_line_if_done and self.progress == self.progress_length:
            print()
    
    def reset(self, print_=False):
        self.progress = 0
        if print_:
            self.print(print_new_line_if_done=False)


if __name__ == "__main__":
    bar = ProgressBar(100, prefix="Name of Progressa dawdawd awd awd", length=100)

    for i in range(100):
        try:
            time.sleep(0.1)
            # bar.increment()
            bar.set_progress(i + 1)
        except KeyboardInterrupt:
            print("\n")
            break
