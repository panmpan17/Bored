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

    def increment(self, print_=True, print_new_line_if_done=False):
        self.progress += 1
        if print_:
            self.print(print_new_line_if_done=print_new_line_if_done)

    def set_progress(self, progress, print_new_line_if_done=False):
        self.progress = progress
        self.print(print_new_line_if_done=print_new_line_if_done)
    
    def export_progress(self):
        length = min(self.max_length, self.length)
        percent = ("{0:." + str(self.decimals) + "f}").format(100 * (self.progress / self.progress_length))
        filled_length = int(length * self.progress // self.progress_length)
        bar = self.fill * filled_length + '-' * (length - filled_length)

        return f'{self.prefix}{bar}| {percent}% {self.suffix}'
    
    def print(self, print_new_line_if_done=False):
        print(self.export_progress())
        sys.stdout.write("\x1b[1A" * (1 + self.prefix_line_count))

        # Print New Line on Complete
        if print_new_line_if_done and self.progress == self.progress_length:
            print()
    
    def reset(self, print_=False):
        self.progress = 0
        if print_:
            self.print(print_new_line_if_done=False)


if __name__ == "__main__":
    bar_1 = ProgressBar(100, prefix=f"Generation 0/100 ", length=100)
    bar_2 = ProgressBar(10, prefix=f"0 Traninng ", length=100)
    top_scores = []

    try:
        for generation in range(100):
            bar_1.set_prefix(f"Generation {generation + 1}/100 ")
            bar_2.set_prefix(f"{bar_1.export_progress()}\nTop Scores: {",".join(top_scores)}\n{generation} Traninng ")
            bar_2.reset(print_=True)

            for i in range(10):
                time.sleep(0.02)
                # bar.increment()
                bar_2.set_progress(i + 1)
            
            top_scores.insert(0, str(generation + 1))
            if len(top_scores) > 5:
                top_scores.pop(-1)
            
            bar_1.increment(print_new_line_if_done=False)

    except KeyboardInterrupt:
        print("\n")
