import matplotlib.pyplot as plt
import numpy as np
import json
import os

from argparse import ArgumentParser



def test():
    #day one, the age and speed of 13 cars:
    x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
    y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])
    plt.scatter(x, y)

    #day two, the age and speed of 15 cars:
    x = np.array([2,2,8,1,15,8,12,9,7,3,11,4,7,14,12])
    y = np.array([100,105,84,105,90,99,90,95,94,100,79,112,91,80,85])
    plt.scatter(x, y)

    plt.show()

class ResultAnalyzer:
    def __init__(self):
        self.scatter_datas = {}

    def read_from_info_json(self, name, info_json_path, trend_window=10):
        with open(info_json_path, "r") as f:
            data = json.load(f)

        scatter_x = []
        scatter_y = []

        avg_x = []
        avg_y = []

        trend_x = []
        trend_y = []

        for history_data in data["history"]:
            generation = history_data["generation"]
            scores = history_data["scores"]
            _sum = 0
            for score in scores:
                scatter_x.append(generation)
                scatter_y.append(score)
                _sum += score

            avg_x.append(generation)
            avg = _sum / len(scores)
            avg_y.append(avg)

            if len(avg_y) >= trend_window:
                trend_x.append(generation)
                trend_y.append(sum(avg_y[len(avg_y) - trend_window:]) / trend_window)

        
        self.scatter_datas[name] = {
            "scatter_x": np.array(scatter_x),
            "scatter_y": np.array(scatter_y),
            "avg_x": np.array(avg_x),
            "avg_y": np.array(avg_y),
            "trend_x": np.array(trend_x),
            "trend_y": np.array(trend_y)
        }
    
    def show(self):
        colors = ["red", "blue", "green", "orange", "purple", "brown"]
        secondary_colors = ["pink", "lightblue", "lightgreen", "yellow", "violet", "tan"]
        count = 0
        for name, data in self.scatter_datas.items():
            color = colors[count % len(colors)]
            plt.scatter(data["scatter_x"],
                        data["scatter_y"],
                        label=name,
                        alpha=0.1,
                        color=secondary_colors[count % len(secondary_colors)],
                        sizes=np.array([1]*len(data["scatter_x"])))
            plt.plot(data["avg_y"], label=name + " avg", color=color)

            if len(data["trend_y"]) > 0:
                plt.plot(data["trend_x"], data["trend_y"], label=name + " trend", color=colors[(count + 1) % len(colors)], linestyle='--')

            count += 1
        
        plt.xlabel("Generation")
        plt.ylabel("Score")
        plt.title("ML Result Analysis")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    parser = ArgumentParser(description="Analyze ML results from info.json files.")

    parser.add_argument("-p", "--path", type=str, help="Path to the info.json file to analyze.",
                        default="output/tic_tac_toe_evolution/info.json")

    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"Error: The file {args.path} does not exist.")
        exit(1)

    # test()
    analyzer = ResultAnalyzer()

    # analyzer.read_from_info_json("first", "output/tic_tac_toe_evolution_1/info.json")
    analyzer.read_from_info_json("second", args.path)
    # analyzer.read_from_info_json("third", "output/tic_tac_toe_evolution_3/info.json")
    
    analyzer.show()
