#------------------------------------------------------------------------

# Pseudocode:
#################### Create class for protfolio holdings
####################### Class perameters of company naem, ticket symbol, industry, buy price, current price, sell price, and possition
#################### define functions for adding stocks to list, presenting the list of stocks in portfolio, and removing stocks from the portfolio
#################### Create a while loop that allows for the user to choose what action they would like to take
### 1. add stock, 2. remove stock, 3. view portfolio, 4. update position, 5. export portfolio, 6. stop using (just cause it to reload option list.)
#################### Create a loop that keeps the programing running, so that all stock inputs are stored.
####################
####################
####################
#------------------------------------------------------------------------
# Program Inputs: company naem, ticket symbol, industry, buy price, current price, sell price, and possition
#####Allow for viewing list, adding, removing, updating, and exporting the data held in the property list.
# Program Outputs: provide list of stocks that are being held in the portfolio
####### Allow for quick updates to portfolio.
#------------------------------------------------------------------------

class Portfolio:

#This will allow the user to define what the values are of the stocks are. Creates a base start
    def __init__(self): 
        self.company_name = ' '
        self.symbol = ' '
        self.industry = ' '
        self.buy_price = 0
        self.current_price = 0
        self.sell_price = 0
        self.position = ' '
        
        

#this is the definition to allow the user to input the stock info
    def add_stock(self):
        self.company_name = input('Enter the company name: ')
        self.symbol = input('Enter the ticker symbol: ')
        self.industry = input('Enter the industry: ')
        self.buy_price = float(input('Enter the price bought at: '))
        self.current_price = float(input('Enter the current market price: '))
        self.sell_price = float(input('Enter the price to sell: '))
        self.position = input('Enter the position of the trade, Long or Short: ')
        
#this will make sure to conver all values into the string value type
    def __str__(self): 
        return('{Name:%s, Symnbol:%s, Industry:%s, Buy_Price:%d, Current_Price:%d, Sell_Price:%d, Position:%s}' %
               (self.company_name, self.symbol, self.industry, self.buy_price, self.current_price, self.sell_price, self.position))

    

#this is an empty list that will allow for a stock to added into the list
stock_list = []




#this will allow the user to edit/update the property in the list, need to see list prior to knowing possition
def update_stock(stock_list):
    stock_id = int(input('Enter the position of the stock in the portfolio: '))
    new_stock = stock.add_stock()
    new_stock = stock.__str__()
    stock_list[stock_id-1] = new_stock #This allows the program to understand which stock in the list is looking to be updated
    print('Stock information updated.')

# this loop will keep the program running so that once live the positions will not be deleted.
# This will also show the user what options they have using this program/database.
user = True
while user:
    print('''
    1. Add a new stock.
    2. Remove a stock.
    3. View portfolio.
    4. Update a position.
    5. Export stock list 
    6. End interaction.
    ''')

    ans = input('What would you to do? Please enter 1 for Add a new stock, 2 to remove a stock ect.')
    if ans == '1':
        stock = Portfolio()
        stock.add_stock()
        stock_list.append(stock.__str__())

    elif ans == '2':
        for i in stock_list:
            stock_list.pop(int(input('Enter the possition of the stock to be removed, use 0 for first stock in the portfolio: ')))
            print('The stock has been removed for the portfolio.')

    elif ans == '3':
        print(stock_list)
    elif ans == '4':
        update_stock(stock_list)
    #This is the code for exporting the txt file the file should be search able on your files under name Portfolio    
    elif ans == '5':
        f = open('Portfolio.txt', 'w')
        f.write(str(stock_list))
        f.close()
    elif ans == '6':
        print('Job complete.')
    else:
        print('Please enter a valid option.')




