from bot import StockBot

def main():
    bot = StockBot()
    
    while True:
        question = input("Ask your question about the stock (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        print(bot.answer(question))

if __name__ == "__main__":
    main()
