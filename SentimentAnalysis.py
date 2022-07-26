from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt

# ask user for ticker
Ticker1 = input('What ticker would you like the analyze ').upper()
Ticker2 = input('What ticker would you like the analyze ').upper()
Ticker3 = input('What ticker would you like the analyze ').upper()
input('Visualize Sentiment Analysis')

# scrape and structure the data
finviz_url = 'https://finviz.com/quote.ashx?t='

tickers = [Ticker1, Ticker2, Ticker3]

news_tables = {}
for ticker in tickers:
    url = finviz_url + ticker

    hdr = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 '
                      'Safari/537.11',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
        'Accept-Encoding': 'none',
        'Accept-Language': 'en-US,en;q=0.8',
        'Connection': 'keep-alive'}

    req = Request(url=url, headers=hdr)
    response = urlopen(req)

    html = BeautifulSoup(response, features='html.parser')
    news_table = html.find(id='news-table')
    news_tables[ticker] = news_table

parsed_data = []

for ticker, news_table in news_tables.items():
    for row in news_table.findAll('tr'):
        title = row.a.text
        date_data = row.td.text.split(' ')

        if len(date_data) == 1:
            time = date_data[0]
        else:
            date = date_data[0]
            time = date_data[1]

        parsed_data.append([ticker, date, time, title])

df = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title'])
df['date'] = pd.to_datetime(df.date).dt.date

# apply sentiment analysis
vader = SentimentIntensityAnalyzer()

f = lambda title: vader.polarity_scores(title)['compound']
df['compound'] = df['title'].apply(f)

# plot the visualization
mean_df = df.groupby(['ticker', 'date']).mean().unstack()
mean_df = mean_df.xs('compound', axis="columns").transpose()
mean_df.plot(kind='bar', figsize=(10, 7))
plt.xlabel('Dates')
plt.ylabel('Composite Sentiment Score')
plt.title('Sentiment Score Analysis')
plt.autoscale()
plt.grid(axis='y', linestyle='-')
plt.show()
