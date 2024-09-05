class Terminal:
    LIGHT_BLUE = '\033[96m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BOLD = '\033[1m'
    GREY = '\033[90m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


if __name__ == "__main__":
    # print(Terminal.LIGHT_BLUE + "Hello, world!" + Terminal.END)
    # print(Terminal.GREEN + "Hello, world!" + Terminal.END)
    # print(Terminal.RED + "Hello, world!" + Terminal.END)
    # print(Terminal.YELLOW + "Hello, world!" + Terminal.END)
    # print(Terminal.BOLD + "Hello, world!" + Terminal.END)
    # print(Terminal.GREY + "Hello, world!" + Terminal.END)
    # print(Terminal.UNDERLINE + "Hello, world!" + Terminal.END)

    for i in range(0, 256):
        print(f"\033[38;5;{i}m{i}", end=" ")
        # print(f"\033[{i}m{i}", end=" ")
        if i % 16 == 0:
            print()
    
    print("\033[0m")
