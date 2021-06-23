# -------------------------------------------------------
# Assignment 2
# Written by Joshua Parial-Bolusan (40063663) Jeffrey Lam(40090989)
# For COMP 472 Section AA – Summer 2021
# --------------------------------------------------------

from dateutil import parser
from bs4 import BeautifulSoup
from bs4.element import ResultSet
from requests import get
import pandas as pd
import re

#Helper Functions that scrape IMDB
def scrapeEpisodes(root, show_url, seasons):

    episodes_container = []

    for season in range(1,seasons+1):

        imdb_url = f"{root}{show_url}/episodes?season={season}"

        response = get(imdb_url)

        html_text = response.text

        soup = BeautifulSoup(html_text, 'html.parser')

        episodes_results: ResultSet = soup.find_all('div', class_='info')

        for episode in episodes_results:
            
            #print(f"{episode.a['title']}, Season: {season}, Link: {root_url}{episode.a['href']}reviews")
            title = episode.a['title']
            link = f"{root}{episode.a['href']}reviews"
            date_raw = episode.find('div', 'airdate').text.strip()
            year = parser.parse(date_raw).year
            #print(year)
            episodes_container.append([title, season, link, year])
            
    return episodes_container


def scrapeEpisodeReviews(episodes_df):

    review_posts = []

    for index, row in episodes_df.iterrows():

        response = get(row["Review Link"])
        soup = BeautifulSoup(response.text, 'html.parser')
        season = row["Season"]
        episode = row["Name"]

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
    


class ImdbService():

    def __init__(self, episodes_df, reviews_df):
        self.episodes_df = episodes_df
        self.reviews_df = reviews_df

    @classmethod
    def from_web(cls, root="https://www.imdb.com", show_url="/title/tt2861424", seasons=4):
        episodes_df = pd.DataFrame( 
            scrapeEpisodes(root, show_url,seasons), 
            columns=['Name', 'Season', 'Review Link', 'Year'])
        reviews_df = pd.DataFrame(
            scrapeEpisodeReviews(episodes_df), columns=['season', 'title', 'rating', 'review']
        ).sample(frac=1)
        return cls(episodes_df, reviews_df)
    
    @classmethod
    def from_csv(cls, csv):
        episodes_df = pd.read_csv(csv)
        reviews_df = pd.DataFrame(
            scrapeEpisodeReviews(episodes_df), columns=['season', 'title', 'rating', 'review']
        ).sample(frac=1)

        return cls(episodes_df, reviews_df) 

    def toCsv(self):
        self.episodes_df.to_csv("data.csv", index=False)
    
