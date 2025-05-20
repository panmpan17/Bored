import time
import csv
import torch
import pandas
import os
import curses
import random
import json

from pprint import pprint
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser


class TicTacToeBoard:
    SYMBOLS = {0: " ", 1: "O", 2: "X", 3: "T"}

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
        if all(x != 0 for x in self.board):
            return 3
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


class GameResultDataset(Dataset):
    @staticmethod
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


class CheckGameResultModel(torch.nn.Module):
    def __init__(self):
        super(CheckGameResultModel, self).__init__()
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
    

class GameResultNeuralNetworkController:
    @property
    def save_file_name(self):
        return "output/tictactoe_model.pth"

    def __init__(self):
        dataset = GameResultDataset()
        self.data_loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        self.model = CheckGameResultModel().to(self.device)

        if os.path.exists(self.save_file_name):
            try:
                self.model.load_state_dict(torch.load(self.save_file_name))
            except Exception as e:
                pass
    
    def start_training(self, train_time: int = 100):
        self.model.train()

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        for epoch in range(train_time):
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

    def __init__(self, network: GameResultNeuralNetworkController, ai_player = None):
        TicTacToeGame.instance = self

        self.network = network
        self.board = TicTacToeBoard()
        self.current_player = 1

        self.selected_index = (0, 0)

        self.stdscr = None

        self.ai_player = ai_player
    
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
        
        start = time.time()
        actual_winner = self.board.get_winner()
        delta = time.time() - start
        self.stdscr.addstr(4, 0, f"Actual Winner: {TicTacToeBoard.SYMBOLS[actual_winner]} ({delta:.2f}s)")

        start = time.time()
        predict_winner = self.network.check_board_result(self.board)
        delta = time.time() - start
        self.stdscr.addstr(5, 0, f"Predict Winner: {TicTacToeBoard.SYMBOLS[predict_winner]} ({delta:.2f}s)")
        
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
            
            if self.ai_player:
                move = self.ai_player.get_move(self.board)
                if move != -1:
                    self.board.set_piece(move, self.current_player)
                    if self.current_player == 1:
                        self.current_player = 2
                    else:
                        self.current_player = 1

        elif key == 27:  # Escape
            self.board.board = [0 for _ in range(9)]
            self.current_player = 1


class DumAIPlayer:
    def __init__(self, player: int):
        self.player = player
    
    def get_move(self, board: TicTacToeBoard):
        indexes = [i for i in range(9) if board.board[i] == 0]
        if len(indexes) == 0:
            return -1
        return random.choice(indexes)


class TicTacToeAIPlayer(torch.nn.Module):
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    def __init__(self, player: int):
        super(TicTacToeAIPlayer, self).__init__()
        # Input data, first is which player the ai is, the following 9 are the board
        self.model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(10, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(128, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(32, 9),
        )
        self.player = player

        self.to(self.device)
    
    def forward(self, x):
        z = self.model(x)
        return torch.nn.functional.log_softmax(z, dim=1)

    def mutate(self, mutate_rate: float = 0.05):
        for param in self.model.parameters():
            if param.requires_grad:
                param.data += torch.randn(param.size()).to(self.device) * mutate_rate

    def get_move(self, board: TicTacToeBoard):
        # print([self.player] + board.board)
        ai_prediction = self.model(torch.tensor([[self.player] + board.board]).float().to(TicTacToeAIPlayer.device))
        return ai_prediction.argmax(1).item()


class AIBattleController:
    def __init__(self, ai_player1: TicTacToeAIPlayer, ai_player2: TicTacToeAIPlayer):
        if ai_player1 is not None:
            ai_player1.player = 1
        if ai_player2 is not None:
            ai_player2.player = 2
        self.ai_player1 = ai_player1
        self.ai_player2 = ai_player2
        self.board = TicTacToeBoard()

        self.current_player = 1
        self.winner = 0

        self.point_for_1 = 0
        self.point_for_2 = 0
    
    def reset(self):
        self.board.board = [0 for _ in range(9)]
        self.current_player = 1
        self.winner = 0
        self.point_for_1 = 0
        self.point_for_2 = 0
    
    def run(self):
        while True:
            if self.current_player == 1:
                move = self.ai_player1.get_move(self.board)
                self.point_for_1 += 10
            else:
                move = self.ai_player2.get_move(self.board)
                self.point_for_2 += 10

            if not self.board.set_piece(move, self.current_player):
                # The move is invalid, the game is over
                if self.current_player == 1:
                    self.point_for_1 -= 5
                else:
                    self.point_for_2 -= 5
                break

            self.winner = self.board.get_winner()
            if self.winner == 1:
                self.point_for_1 += 100
                break
            elif self.winner == 2:
                self.point_for_2 += 100
                break
            elif self.winner == 3: # draw
                self.point_for_1 += 10
                # Accommadate player 2 move is fewer
                self.point_for_2 += 20
                break

            if self.current_player == 1:
                self.current_player = 2
            else:
                self.current_player = 1


class EvolutionTraining:
    def __init__(self, population_size: int = 100, top_population = 5, evaluate_game_count=20, mutation_rate: float = 0.05):
        self.population_size = population_size
        self.top_population = top_population
        self.mutation_rate = mutation_rate
        self.evaluate_game_count = evaluate_game_count

        self.scores = []
        self.population = []
        self.battle_controller = AIBattleController(None, DumAIPlayer(2))
        self.generation = 1

        self.history = []

    def init_population(self):
        files = []
        for generation_data in self.history:
            if "files" in generation_data:
                files = generation_data["files"]
        
        if len(files) > 0:
            print(f"Loading population from {files}")
            
            folder = os.path.dirname(self.json_info_file)
            mutate_per_top_population = self.population_size // self.top_population
            for file in files:
                ai = TicTacToeAIPlayer(1)
                ai.load_state_dict(torch.load(os.path.join(folder, file)))
                self.population.append(ai)

                for i in range(mutate_per_top_population - 1):
                    new_ai = TicTacToeAIPlayer(ai.player)
                    new_ai.load_state_dict(ai.state_dict())
                    new_ai.mutate(self.mutation_rate)
                    self.population.append(new_ai)
            
        else:
            print("Loading population from scratch")
            self.init_random_population()

    def init_random_population(self):
        self.population = [TicTacToeAIPlayer(i) for i in range(self.population_size)]
    
    def load_info(self):
        if os.path.exists(self.json_info_file):
            with open(self.json_info_file, "r") as f:
                data = json.load(f)
                self.generation = data["generation"]
                self.history = data["history"]
    
    def dump_info(self):
        if not os.path.exists("output/tic_tac_toe_evolution"):
            os.makedirs("output/tic_tac_toe_evolution")
        
        data = {}
        data["generation"] = self.generation
        data["history"] = self.history

        with open(self.json_info_file, "w") as f:
            json.dump(data, f, indent=4)
    
    @property
    def json_info_file(self):
        return f"output/tic_tac_toe_evolution/info.json"

    def evaluate_population(self):
        self.scores = []
        for i, ai in enumerate(self.population):
            avg_score = self.evaluate_one_ai(ai)
            self.scores.append((i, avg_score))
        
        print(f"Generation {self.generation} completed, evaluating population...")
        self.scores.sort(key=lambda x: x[1], reverse=True)
        print("Top population scores:", ",".join([str(score) for _, score in self.scores[:self.top_population]]))

        generation_data = {}
        generation_data["generation"] = self.generation
        generation_data["scores"] = [score for _, score in self.scores]

        generation_data["params"] = {}
        generation_data["params"]["population_size"] = self.population_size
        generation_data["params"]["top_population"] = self.top_population
        generation_data["params"]["mutation_rate"] = self.mutation_rate
        generation_data["params"]["evaluate_game_count"] = self.evaluate_game_count

        self.history.append(generation_data)

    def next_generation(self):
        scores = self.scores[:self.top_population]
        
        new_population = []

        mutate_per_top_population = self.population_size // self.top_population
        for i, _ in scores:
            ai = self.population[i]
            new_population.append(ai)

            for i in range(mutate_per_top_population - 1):
                new_ai = TicTacToeAIPlayer(ai.player)
                new_ai.load_state_dict(ai.state_dict())
                new_ai.mutate(self.mutation_rate)
                new_population.append(new_ai)
        
        for i in range(self.population_size - len(new_population)):
            random_ai = self.population[random.choice(scores)[0]]

            new_ai = TicTacToeAIPlayer(random_ai.player)
            new_ai.load_state_dict(random_ai.state_dict())
            new_ai.mutate(self.mutation_rate)
            new_population.append(new_ai)
        
        print(f"New population size: {len(new_population)} readying for next generation")
        self.population = new_population
        self.generation += 1
        self.scores.clear()

    def evaluate_one_ai(self, ai: TicTacToeAIPlayer):
        self.battle_controller.reset()
        self.battle_controller.ai_player1 = ai

        scores = []
        for i in range(self.evaluate_game_count):
            self.battle_controller.run()
            scores.append(self.battle_controller.point_for_1)
            self.battle_controller.reset()

        return sum(scores) / self.evaluate_game_count

    def save_top_population(self):
        if len(self.scores) == 0:
            return
        
        folder = os.path.dirname(self.json_info_file)
        files = []
        for i, score in self.scores[:self.top_population]:
            ai = self.population[i]
            file_name = f"ai_{self.generation}_{i}.pth"
            files.append(file_name)
            ai_file = os.path.join(folder, file_name)
            torch.save(ai.state_dict(), ai_file)
        
        self.history[-1]["files"] = files


if __name__ == "__main__":
    parser = ArgumentParser()

    sub = parser.add_subparsers(dest="command")

    train_parser = sub.add_parser("train", help="Train the model")
    train_parser.add_argument("-t", "--time", type=int, default=100, help="Time to train the model")

    evolution = sub.add_parser("evolution", help="Train the model using evolution")
    evolution.add_argument("-p", "--population", type=int, default=100, help="Population size")
    evolution.add_argument("-t", "--top", type=int, default=5, help="Top population size")
    evolution.add_argument("-m", "--mutation", type=float, default=0.05, help="Mutation rate")
    evolution.add_argument("-e", "--evaluate", type=int, default=20, help="Evaluate game count")
    evolution.add_argument("-g", "--generation", type=int, default=10, help="Generation count")

    test_parser = sub.add_parser("test", help="Test the model")

    game_parser = sub.add_parser("game", help="Test the model")

    args = parser.parse_args()

    if args.command == "train":
        controller = GameResultNeuralNetworkController()
        controller.evaluate_accuracy()
        controller.start_training(args.time)
        controller.evaluate_accuracy()
    
    elif args.command == "evolution":
        training = EvolutionTraining(population_size=args.population, top_population=args.top,
                                    mutation_rate=args.mutation, evaluate_game_count=args.evaluate)

        training.load_info()
        training.init_population()
        
        for i in range(args.generation):
            training.evaluate_population()
            if i != args.generation - 1:
                training.next_generation()
        
        training.save_top_population()
        training.dump_info()
    
    elif args.command == "test":
        board = TicTacToeBoard()
        ai = TicTacToeAIPlayer(1)

        battle_controller = AIBattleController(ai, DumAIPlayer(2))
        for i in range(50):
            battle_controller.run()
            battle_controller.board.pretty_print()
            print(f"Winner: {TicTacToeBoard.SYMBOLS[battle_controller.winner]}, Points 1: {battle_controller.point_for_1}, Points 2 {battle_controller.point_for_2}")
            battle_controller.reset()
    
    elif args.command == "game":
        controller = GameResultNeuralNetworkController()

        dum_ai = DumAIPlayer(2)
        test_ai = TicTacToeAIPlayer(1)
        test_ai.load_state_dict(torch.load("output/tic_tac_toe_evolution/ai_117_84.pth"))

        game = TicTacToeGame(controller, test_ai)
        curses.wrapper(TicTacToeGame.static_run)




