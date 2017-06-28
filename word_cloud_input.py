#!/usr/bin/env python3
import os
import logging
import csv
from pathlib import Path
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import codecs
import re

path_to_directory = os.path.join(os.path.expanduser('~'), 'Desktop')+"/Test"

print(path_to_directory)
file_content = []

for filename in os.listdir(path_to_directory):

	if filename.endswith(".txt"):
		#print(path_to_directory+"/"+filename)
	 	f=codecs.open(path_to_directory+"/"+filename,'r+','utf-8')
		lines=f.read()
		lines = re.sub('\W+',' ',lines)
		f.close()
		file_content.append(lines)
		file_content.append('\nBREAKS HERE ')
		file_content.append('\n')


file_content2 =''.join(file_content)
output_file = open("/Users/user_nameteam/Desktop/clustering/clustering_test.txt","w")
output_file.write(file_content2)
#print("Word cloud input generated")

