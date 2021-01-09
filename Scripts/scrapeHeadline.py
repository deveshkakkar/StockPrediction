from selenium import webdriver
import pandas as pd
import time
ticker = "tsla"
website = 'http://www.nasdaq.com/symbol/' + ticker + '/news-headlines'

def getText(someList):
    returnList = []
    for i in someList:
        returnList.append(i.text)
    return returnList

dr = webdriver.Chrome('/usr/local/bin/chromedriver')
dr.get(website)
titles=[]
dates=[]
next = dr.find_elements_by_class_name('pagination__next')

while(next[0].is_enabled()):
    titles.append(getText(dr.find_elements_by_class_name('quote-news-headlines__item-title')))
    dates.append(getText(dr.find_elements_by_class_name('quote-news-headlines__date')))
    dr.execute_script("arguments[0].click();", next[0])
    time.sleep(3)

table = pd.DataFrame([titles,dates])
table.to_csv("Headlines.csv")
