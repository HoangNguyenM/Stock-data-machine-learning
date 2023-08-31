import requests
from bs4 import BeautifulSoup

def print_SP500_symbols():
    # URL of the Wikipedia page containing the list of S&P 500 companies
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    # Send a GET request to the URL
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the table containing the list of S&P 500 companies
    table = soup.find('table', {'class': 'wikitable sortable'})

    # Initialize an empty list to store the symbols
    symbols = []

    # Iterate through the rows of the table and extract the symbols from the first column
    for row in table.find_all('tr')[1:]:
        columns = row.find_all('td')
        symbol = columns[0].text.strip()
        symbols.append(symbol)

    # Write the symbols to a text file
    output_file = "utils/SP500.txt"
    with open(output_file, 'w') as f:
        for symbol in sorted(symbols):
            f.write(symbol + '\n')

def get_SP500_symbols():
    # Read the symbols from the text file into a list
    ticker_file = "utils/SP500.txt"
    ticker_list = []

    with open(ticker_file, 'r') as f:
        for line in f:
            symbol = line.strip()
            ticker_list.append(symbol)

    return ticker_list

def print_SP100_symbols():
    # URL of the Wikipedia page containing the list of S&P 100 companies
    url = "https://en.wikipedia.org/wiki/S%26P_100"

    # Send a GET request to the URL
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the table containing the list of S&P 100 companies
    table = soup.find('table', {'class': 'wikitable sortable'})

    # Initialize an empty list to store the symbols
    symbols = []

    # Iterate through the rows of the table and extract the symbols from the first column
    for row in table.find_all('tr')[1:]:
        columns = row.find_all('td')
        symbol = columns[0].text.strip()
        symbols.append(symbol)

    # Write the symbols to a text file
    output_file = "utils/SP100.txt"
    with open(output_file, 'w') as f:
        for symbol in sorted(symbols):
            f.write(symbol + '\n')

def get_SP100_symbols():
    # Read the symbols from the text file into a list
    ticker_file = "utils/SP100.txt"
    ticker_list = []

    with open(ticker_file, 'r') as f:
        for line in f:
            symbol = line.strip()
            ticker_list.append(symbol)

    return ticker_list

# print_SP100_symbols()
# print_SP500_symbols()