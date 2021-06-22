# -------------------------------------------------------
# Assignment 2
# Written by Joshua Parial-Bolusan (40063663) Jeffrey Lam(40090989)
# For COMP 472 Section (your lab section) – Summer 2021
# --------------------------------------------------------

from dateutil import parser
from bs4 import BeautifulSoup
from bs4.element import ResultSet
from requests import get
import pandas as pd
import re

class ImdbService():
    root_url = "https://www.imdb.com"

    def __init__(self, show_endpoint="/title/tt2861424", seasons=4) -> None:
        self.show_endpoint =f"{self.root_url}{show_endpoint}"
        self.episodes_df = pd.DataFrame( 
            self.__scrapeEpisodes(seasons=seasons), 
            columns=['title', 'season', 'review_link', 'year'])
        self.reviews_df = pd.DataFrame(
            self.__scrapeEpisodeReviews(), columns=['season', 'title', 'rating', 'review']
        ).sample(frac=1)
    
    def getReviews(self, season, episode):
        pass

    def toCsv(self):
        self.episodes_df.to_csv("out.csv", index=False)
    
    def __scrapeEpisodes(self, seasons):

        episodes_container = []

        for season in range(1,seasons+1):

            imdb_url = f"{self.show_endpoint}/episodes?season={season}"

            response = get(imdb_url)

            html_text = response.text

            soup = BeautifulSoup(html_text, 'html.parser')

            episodes_results: ResultSet = soup.find_all('div', class_='info')

            for episode in episodes_results:
                
                #print(f"{episode.a['title']}, Season: {season}, Link: {root_url}{episode.a['href']}reviews")
                title = episode.a['title']
                link = f"{self.root_url}{episode.a['href']}reviews"
                date_raw = episode.find('div', 'airdate').text.strip()
                year = parser.parse(date_raw).year
                #print(year)
                episodes_container.append([title, season, link, year])
                
        return episodes_container


    def __scrapeEpisodeReviews(self):

        review_posts = []

        for index, row in self.episodes_df.iterrows():

            response = get(row["review_link"])
            soup = BeautifulSoup(response.text, 'html.parser')
            season = row["season"]
            episode = row["title"]

            review_list = soup.find_all('div', class_='review-container')
            for review in review_list:

                
                review_block = review.find('div', class_='ipl-ratings-bar')
                if review_block == None:
                    continue
                rating = int(review_block.find('span').text.strip().split("/")[0])
                #print(rating)
                review_text = review.find('div', class_='text show-more__control').text.lower()
                review_text = re.sub(r'[\t\n\.]', '', review_text)
                review_label = 'negative'

                review_title = review.find('a', class_="title").text.strip()

                if rating >= 8:
                    review_label = 'positive'
                #print(review_block)
                review_posts.append([season, review_title, review_label, review_text])
        
        return review_posts
    