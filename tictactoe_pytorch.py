import csv
import torch
import pandas

from pprint import pprint
from torch.utils.data import Dataset


class TicTacToeBoard:
    SYMBOLS = {0: " ", 1: "X", 2: "O"}

    def __init__(self, board_string: str = None):
        self.board = [0 for _ in range(9)]

        if board_string:
            self.read_from_string(board_string)
    
    def __repr__(self):
        return f"{self.SYMBOLS[self.board[0]]}{self.SYMBOLS[self.board[1]]}{self.SYMBOLS[self.board[2]]}{self.SYMBOLS[self.board[3]]}{self.SYMBOLS[self.board[4]]}{self.SYMBOLS[self.board[5]]}{self.SYMBOLS[self.board[6]]}{self.SYMBOLS[self.board[7]]}{self.SYMBOLS[self.board[8]]}"
    
    def read_from_string(self, board_string: str):
        for i, char in enumerate(board_string):
            if char == "X" or char == "x" or char == "1":
                self.board[i] = 1
            elif char == "O" or char == "o" or char == "2":
                self.board[i] = 2
            else:
                self.board[i] = 0

    def set_piece(self, position: int | tuple[int, int], player: int):
        if isinstance(position, tuple):
            position = position[0] * 3 + position[1]

        if self.board[position] == 0:
            self.board[position] = player
            return True

        return False 

    def pretty_print(self):
        print(f"{self.SYMBOLS[self.board[0]]}{self.SYMBOLS[self.board[1]]}{self.SYMBOLS[self.board[2]]}")
        print(f"{self.SYMBOLS[self.board[3]]}{self.SYMBOLS[self.board[4]]}{self.SYMBOLS[self.board[5]]}")
        print(f"{self.SYMBOLS[self.board[6]]}{self.SYMBOLS[self.board[7]]}{self.SYMBOLS[self.board[8]]}")
    
    def get_winner(self):
        winning_positions = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8), # rows
            (0, 3, 6), (1, 4, 7), (2, 5, 8), # columns
            (0, 4, 8), (2, 4, 6)            # diagonals
        ]
        for a, b, c in winning_positions:
            if self.board[a] == self.board[b] == self.board[c] != 0:
                return self.board[a]
        return 0

    def to_csv_row(self):
        return [x for x in self.board] + [self.get_winner()]


def create_trainning_data():
    boards = []

    def recursive_all_possible_boards(prevboard_string: str, depth: int = 0):
        global boards

        for choice in range(3):
            if choice == 0:
                new_board_string = prevboard_string + "0"
            elif choice == 1:
                new_board_string = prevboard_string + "1"
            else:
                new_board_string = prevboard_string + "2"

            if len(new_board_string) == 9:
                boards.append(TicTacToeBoard(new_board_string))
            else:
                recursive_all_possible_boards(new_board_string, depth + 1)

    recursive_all_possible_boards("")

    writer = csv.writer(open("tictactoe_boards.csv", "w"))
    writer.writerow(["a", "b", "c", "d", "e", "f", "g", "h", "i", "winner"])
    for board in boards:
        writer.writerow(board.to_csv_row())


class CustomStarDataset(Dataset):
    # This loads the data and converts it, make data rdy
    def __init__(self):
        # load data
        self.df = pandas.read_csv("tictactoe_boards.csv")
        # extract labels
        self.df_labels = self.df[["winner"]]
        # drop non numeric columns to make tutorial simpler, in real life do categorical encoding
        self.df = self.df.drop(columns=["winner"])
        # conver to torch dtypes
        self.dataset = torch.tensor(self.df.to_numpy()).float()

        self.labels = torch.tensor(self.df_labels.to_numpy().reshape(-1)).long()
    
    # This returns the total amount of samples in your Dataset
    def __len__(self):
        return len(self.dataset)
    
    # This returns given an index the i-th sample and label
    def __getitem__(self, idx):
        return self.dataset[idx],self.labels[idx]


dataset = CustomStarDataset()
