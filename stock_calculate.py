import math

from datetime import datetime
from argparse import ArgumentParser


USA_STOCK_TAX = 0.001


def caulculate_usa_stock(stock_price: float, amount: int, currency_exchange: float):
    stock_need_usa = round(stock_price * amount, 2)
    tax_usa = round(stock_need_usa * USA_STOCK_TAX, 2)
    usa_total = round(stock_need_usa + tax_usa, 2)

    stock_need_tw = math.ceil(usa_total * currency_exchange)
    tax_tw = math.ceil(tax_usa * currency_exchange)
    total = math.ceil(stock_need_tw + tax_tw)

    return stock_need_usa, tax_usa, usa_total, stock_need_tw, tax_tw, total

def print_usa_stock(stock_price: float, amount: int, currency_exchange: float):
    stock_need_usa, tax_usa, usa_total, stock_need_tw, tax_tw, total = caulculate_usa_stock(stock_price, amount, currency_exchange)

    print(f"Stock price: {stock_price}")
    print(f"Stock amount: {amount}")
    print(f"Currency exchange rate: {currency_exchange}")
    print("--------------------")
    print(f"USA stock need: {stock_need_usa}")
    print(f"USA tax: {tax_usa}")
    print(f"USA total: {usa_total}")
    print("--------------------")
    print(f"TW stock need: {stock_need_tw}")
    print(f"TW tax: {tax_tw}")
    print(f"Total: {total}")

def print_usa_time():
    now = datetime.now()
    if now.month >= 3 and now.month <= 11:
        # 21:30-04:00
        print("USA stock open time: 21:30-04:00")
        if now.hour < 4 or now.hour >= 21:
            print("USA stock market is open")
        else:
            print("USA stock market is close")
    
    else:
        # 22:30-05:00
        print("USA stock open time: 22:30-05:00")
        if now.hour < 5 or now.hour >= 22:
            print("USA stock market is open")
        else:
            print("USA stock market is close")

if __name__ == '__main__':
    parser = ArgumentParser(description='Stock Calculate')
    
    sub = parser.add_subparsers(dest='command')

    usa_parser = sub.add_parser("usa", help="USA stock calculate")
    usa_parser.add_argument("stock_price", type=float, help="Stock price")
    usa_parser.add_argument("amount", type=int, help="Stock amount")
    usa_parser.add_argument("currency_exchange", type=float, help="Curency exchange rate")

    time_parser = sub.add_parser("usa-time", help="Time calculate")


    args = parser.parse_args()

    if args.command == 'usa':
        print_usa_stock(args.stock_price, args.amount, args.currency_exchange)
        
    elif args.command == 'usa-time':
        print_usa_time()

