from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import os
from PIL import Image
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import numpy as np
from pathlib import Path

os.chdir(os.path.expanduser('~/Desktop/Escalations/'))
#f = open("test_cloud.txt","r")
contents = (Path('/Users/Activity/Desktop/Escalations/Output.txt').read_text()).lower()
#contents = f.read()
#f.close()
example_text = word_tokenize(contents)
stop_words = set(stopwords.words("english"))

new_words = set([,words])
#print(example_text)
user_name_logo = np.array(Image.open('/Users/Activity/Desktop/final_logo_ex.jpg'))
#removing punc
example_text_new = list(filter(lambda x: x not in string.punctuation, example_text))

#print(example_text_new)

#remove stopwords
cleaned_text_new = list(filter(lambda x: x not in stop_words, example_text_new))
cleaned_text = list(filter(lambda x: x not in new_words, cleaned_text_new))
#text = [word for word in contents if word not in stopwords.words('english')]
#print(cleaned_text)

cleaned_text =' '.join(cleaned_text)
#print(cleaned_text)
#print(type(cleaned_text))
more_stopwords = {words}
STOPWORDS = STOPWORDS.union(more_stopwords)
wc = WordCloud(background_color="black",max_words=200,mask=user_name_logo,stopwords=STOPWORDS)

wc.generate(cleaned_text)

wc.to_file('/Users/Activity/Desktop/Escalations/word_cloud_user_name_test.png')

print("Wordcloud generated")
