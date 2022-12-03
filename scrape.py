import numpy as np
import pandas as pd
from selenium import webdriver 
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from scrapy import Selector
from googletrans import Translator
import time
from tqdm import tqdm
import warnings


warnings.filterwarnings("ignore")

translator = Translator()
chromeDriver = webdriver.Chrome('chromedriver.exe')
url = 'https://www.imdb.com/title/tt0137523/reviews/?ref_=tt_ql_urv'
time.sleep(1)
chromeDriver.get(url)

time.sleep(1)
print(chromeDriver.title)
time.sleep(1)

body = chromeDriver.find_element(By.CSS_SELECTOR, 'body')
body.send_keys(Keys.PAGE_DOWN)
time.sleep(1)
body.send_keys(Keys.PAGE_DOWN)
time.sleep(1)
body.send_keys(Keys.PAGE_DOWN)

selector = Selector(text = chromeDriver.page_source)
review_counts = selector.css('.lister .header span::text').extract_first().replace(',','').split(' ')[0]
print(review_counts)
pages = int(int(review_counts)/25)
print(pages)


for i in range(pages):
    try:
        css_selector = 'load-more-trigger'
        time.sleep(1)
        body = chromeDriver.find_element(By.CSS_SELECTOR, 'body')
        body.send_keys(Keys.PAGE_DOWN)
        time.sleep(1)
        body.send_keys(Keys.PAGE_DOWN)
        time.sleep(1)
        body.send_keys(Keys.PAGE_DOWN)
        chromeDriver.find_element(By.ID, css_selector).click()
        print("Clicked load more")
    except:
        pass

ratingList = []
reviewList = []
errorList = []
errorMsg = []


reviews = chromeDriver.find_elements(By.CSS_SELECTOR, 'div.review-container')
for d in tqdm(reviews):
    try:
        sel2 = Selector(text = d.get_attribute('innerHTML'))
        try:
            rating = sel2.css('.rating-other-user-rating span::text').extract_first()
        except:
            rating = 11
        try:
            review = sel2.css('.text.show-more__control::text').extract_first()
        except:
            review = np.NaN
        if int(rating) <= 5 and review!="":
            translated = translator.translate(review).text
            ratingList.append(0)
            reviewList.append(translated)
        if int(rating) > 5 and int(rating)<=10 and review!="":
            translated = translator.translate(review).text
            ratingList.append(1)
            reviewList.append(translated)

    except Exception as e:
        errorList.append(url)
        errorMsg.append(e)


review_df = pd.DataFrame({
    'Rating':ratingList,
    'Review':reviewList,
    })


print(review_df)
review_df.to_csv('dataset.csv')