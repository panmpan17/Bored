import csv
import torch
import pandas
import os
import curses

from pprint import pprint
from torch.utils.data import Dataset, DataLoader


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
    
    def get_line_with_select_symbol(self, selected_index: tuple[int, int]):
        lines = []

        for i in range(3):
            line = ""
            for j in range(3):
                if (i, j) == selected_index:
                    line += f"[{self.SYMBOLS[self.board[i * 3 + j]]}]"
                else:
                    line += f" {self.SYMBOLS[self.board[i * 3 + j]]} "
            lines.append(line)
        
        return lines


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
        return self.dataset[idx], self.labels[idx]


class NeuralNetworkModel(torch.nn.Module):
    def __init__(self):
        super(NeuralNetworkModel, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(9, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(128, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(32, 3),
        )

    def forward(self, x):
        z = self.model(x)
        return torch.nn.functional.log_softmax(z, dim=1)
    

class NeuralNetworkController:
    @property
    def save_file_name(self):
        return "tictactoe_model.pth"

    def __init__(self):
        dataset = CustomStarDataset()
        self.data_loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        self.model = NeuralNetworkModel().to(self.device)

        if os.path.exists(self.save_file_name):
            try:
                self.model.load_state_dict(torch.load(self.save_file_name))
            except Exception as e:
                pass
    
    def start_training(self):
        self.model.train()

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        for epoch in range(20):
            for batch, label in self.data_loader:
                X, y = batch.to(self.device), label.to(self.device)

                optimizer.zero_grad()
                output = self.model(X)
                # print(X)
                # print(output)
                # print(y)
                loss = loss_fn(output, y)
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch + 1} completed")

        torch.save(self.model.state_dict(), "tictactoe_model.pth")

    def evaluate_accuracy(self):
        loss_fn = torch.nn.CrossEntropyLoss()

        size = len(self.data_loader.dataset)
        num_batches = len(self.data_loader)
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in self.data_loader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    def check_board_result(self, board: TicTacToeBoard):
        ai_prediction = self.model(torch.tensor([board.board]).float().to(self.device))
        return ai_prediction.argmax(1).item()
    
    def test_data(self):
        result = self.model(torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0]]).float().to(self.device))
        print(result.argmax(1).item())

        result = self.model(torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0, 0]]).float().to(self.device))
        print(result.argmax(1))

        result = self.model(torch.tensor([[2, 2, 2, 0, 0, 0, 0, 0, 0]]).float().to(self.device))
        print(result.argmax(1))


class TicTacToeGame:
    instance = None

    @staticmethod
    def static_run(stdscr):
        TicTacToeGame.instance.run(stdscr)

    def __init__(self, network: NeuralNetworkController):
        TicTacToeGame.instance = self

        self.network = network
        self.board = TicTacToeBoard()
        self.current_player = 1

        self.selected_index = (0, 0)

        self.stdscr = None
    
    def run(self, stdscr):
        self.stdscr = stdscr

        self.draw_board()

        while True:
            try:
                key = stdscr.getch()
            except KeyboardInterrupt:
                break

            self.handle_input(key)
            self.draw_board()
    
    def draw_board(self):
        self.stdscr.clear()

        lines = self.board.get_line_with_select_symbol(self.selected_index)
        for i, line in enumerate(lines):
            self.stdscr.addstr(i, 0, line)
        
        actual_winner = self.board.get_winner()
        self.stdscr.addstr(4, 0, f"Actual Winner: {TicTacToeBoard.SYMBOLS[actual_winner]}")

        predict_winner = self.network.check_board_result(self.board)
        self.stdscr.addstr(5, 0, f"Predict Winner: {TicTacToeBoard.SYMBOLS[predict_winner]}")
        
        self.stdscr.refresh()
    
    def handle_input(self, key: int):
        if key == 258:  # Down
            self.selected_index = (self.selected_index[0] + 1, self.selected_index[1])
            if self.selected_index[0] > 2:
                self.selected_index = (0, self.selected_index[1])
        elif key == 259:  # Up
            self.selected_index = (self.selected_index[0] - 1, self.selected_index[1])
            if self.selected_index[0] < 0:
                self.selected_index = (2, self.selected_index[1])
        elif key == 260:  # Left
            self.selected_index = (self.selected_index[0], self.selected_index[1] - 1)
            if self.selected_index[1] < 0:
                self.selected_index = (self.selected_index[0], 2)
        elif key == 261:  # Right
            self.selected_index = (self.selected_index[0], self.selected_index[1] + 1)
            if self.selected_index[1] > 2:
                self.selected_index = (self.selected_index[0], 0)
        elif key == 10:  # Enter
            self.board.set_piece(self.selected_index, self.current_player)
            if self.current_player == 1:
                self.current_player = 2
            else:
                self.current_player = 1

    


if __name__ == "__main__":
    controller = NeuralNetworkController()
    controller.evaluate_accuracy()
    controller.start_training()
    controller.evaluate_accuracy()
    # controller.test_data()

    # game = TicTacToeGame(controller)
    # curses.wrapper(TicTacToeGame.static_run)




