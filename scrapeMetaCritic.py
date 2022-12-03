import requests
from bs4 import BeautifulSoup
import pandas as pd
from googletrans import Translator

rating_review = {'Rating': [], 'Review': []}

movie_name = "the-godfather-full"

for page in range(0, 20):
    print(page)
    url = 'https://www.metacritic.com/movie/the-godfather/user-reviews?page=' + str(page)
    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    soup = BeautifulSoup(response.text, 'html.parser')

    for review in soup.find_all('div', {'class': 'review pad_top1'}):
        if review.find('span', class_='author') is None:
            break

        rating = int((review.find('div', {'class': 'left fl'}).find_all('div')[0]).text)
        if rating <= 5:
            rating_review['Rating'].append(int(0))
        if 5 < rating <= 10:
            rating_review['Rating'].append("")
        try:
            if review.find('span', class_='blurb blurb_expanded'):
                rating_review['Review'].append(review.find('span', {'class': 'blurb blurb_expanded'}).text)
            else:
                rating_review['Review'].append(review.find('div', class_='review_body').find('span').text)
        except:
            rating_review['Review'].append("")

    movie_reviews = pd.DataFrame(rating_review)


# cleaning
print(movie_reviews.shape)

new_data = movie_reviews.dropna()
new_data = new_data.reset_index(drop=True)


translator = Translator()

translated_reviews = {'Rating': [], 'Review': []}

new_data = pd.DataFrame(new_data)

for i in range(0, len(new_data['Review'])):
    translated = translator.translate(new_data['Review'][i]).text
    translated_reviews['Review'].append(translated)
    translated_reviews['Rating'].append(new_data['Rating'][i])
    print(i)

reviews = pd.DataFrame(translated_reviews)
reviews.to_csv(movie_name + "reviews.csv")
