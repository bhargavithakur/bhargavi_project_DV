#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 22:48:20 2022

@author: bhargavithakur
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 19:03:33 2021

@author: bhargavithakur
"""


#Studyig Domestic Homicide and Abuse in the US


#importing all useful libraries 

import pandas as pd
import matplotlib.pyplot as plt
import geopandas
import os
import tabula
from tabula import read_pdf
import PyPDF2
import requests
import numpy as np
import pdfrw
from pdfrw import PdfReader, PdfWriter
import spacy 
import autocorrect
from spacy import displacy
import re
import string
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
import nltk
from nltk.corpus import stopwords, brown
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from autocorrect import spell
from autocorrect import Speller
nltk.download('punkt')
from nltk import ngrams



#Part1: 
#Using Chorpleth to visualize and explore the percentage of men and women that face domestic abuse in 
#different states of the US. 

#will need path to be changed 
path = r'/Users/bhargavithakur/Desktop/data and programming 2 /final project/'
#Using python to download csv document showing data of percentage of women and men facing domestic abuse 

#will require the csv path to be changed
df_us_domestic_abuse = pd.read_csv('/Users/bhargavithakur/Desktop/data and programming 2 /final project/csvData (1).csv')

#downloading the shapefile of the US
us_states = os.path.join(path, 'cb_2018_us_state_500k', 'cb_2018_us_state_500k.shp')
df_us_states = geopandas.read_file(us_states)
df_us_states = df_us_states.rename(columns={'NAME':'State'})

#Merging the domestic abuse data with the shapefile data of the US

df_merged = df_us_states.merge(df_us_domestic_abuse , on = 'State')
variables = ['percWomen' , 'percMen']


#Exploring through interactive map the percentage of men and women facing domestic abuse in the US
def map_explore_usa(variable , cmap = True):
    for variable in variables: 
        map_explore_usa = df_merged.explore(variables[1] , cmap = 'YlGnBu')
        return map_explore_usa
map_explore_usa('percMen' , cmap = True)
#can explore values similarly for percWomen to view the distribution of domestic abuse acorss different states of the USA

#Part2: 
#downloading the original file which give statistics on domestic abuse in the us from 2003–2012

#will require changing the file_location
pdf_trend = r"/Users/bhargavithakur/Desktop/data and programming 2 /final project/ndv0312 (1).pdf"
#Parsing the PDF Document 
#Retriving tables on Rate and percentage of violent victimization, by victim–offender relationship, 2003–2012 and converting 
#it into a csv document
df_trend_2 = tabula.read_pdf(pdf_trend , pages = '5')
df_victim_relationship = df_trend_2[0]
df_victim_relationship.to_csv("victim_relationship.csv")


#will require changing the file_location
new_fname = r"/Users/bhargavithakur/Desktop/data and programming 2 /final project/victim_relationship.csv"
df_victim_relation = pd.read_csv(os.path.join(path, new_fname))
df_victim_relation = df_victim_relation.drop(['Unnamed: 0'] , axis = 1)
df_victim_relation = df_victim_relation.rename(columns={"violent crime": "Total Violent Crime", "violent crimea": "Serious Violent Crime" })
#working with only the rates of domestic abuse, and  thus dropping the percentage variables and data 
df_violence_rate = df_victim_relation.drop({'Victim–offender relationship.1',  'Male', 'Female', 'Male.1', 'Female.1'}, axis=1)
#df_violence_rate
#Summarizing the data on  domestic violence cases by type of violence (a) Serious Violence Crime and (b) Assault 
#by looking at the relation of the 'victim' with the abuser, using plots. 
fig, ax = plt.subplots()
ax.bar(df_violence_rate['Victim–offender relationship'] ,df_violence_rate['assault'],  label='assault')
ax.bar(df_violence_rate['Victim–offender relationship'] , df_violence_rate['Serious Violent Crime'], label='Serious Violent Crime')
ax.set_xticklabels(df_violence_rate['Victim–offender relationship'] , fontsize= 5 ,rotation= 90)
ax.set_title('Violence Rate by Relationship to Victim')
ax.legend()
ax.autoscale_view()
plt.show()
#saving the plot figure 
fig.savefig('ax_domestic_abuse_victim_relation.png')



#Part3
#In this part I analyze the harshness of the firearm protection laws in four states 
#Four States: 1. Alabama 2. Florida 3. New York 4. California

#downloading the four files which have laws stating for the respective four states

#path1, path2, path3, path4 will require change of path
path1 = r"/Users/bhargavithakur/Desktop/data and programming 2 /final project/alabama/ncadv_alabama_fact_sheet_2020.pdf"
alabama_law = PyPDF2.PdfFileReader(path1)
path2 = r"/Users/bhargavithakur/Desktop/data and programming 2 /final project/FLORIDA/ncadv_florida_fact_sheet_2020.pdf"
florida_law = PyPDF2.PdfFileReader(path2)
path3 = r"/Users/bhargavithakur/Desktop/data and programming 2 /final project/newyork/ncadv_new_york_fact_sheet_2020.pdf"
newyork_law = PyPDF2.PdfFileReader(path3)
path4 = r"/Users/bhargavithakur/Desktop/data and programming 2 /final project/california/ncadv_california_fact_sheet_2020.pdf"
cali_law = PyPDF2.PdfFileReader(path4)

#defining the useful pages in each document
pages = (0,1)
#getting texts from all the four documents
text_alabama = []
for pnum in pages:
    page = alabama_law.getPage(pnum)
    text_alabama.append(page.extractText())
alabama_text = text_alabama[0]


text_florida = []
for pnum in pages:
    page = florida_law.getPage(pnum)
    text_florida.append(page.extractText())
florida_text = text_florida[0]

text_newyork = []
for pnum in pages:
    page = newyork_law.getPage(pnum)
    text_newyork.append(page.extractText())
ny_text_0 = text_newyork[0]
ny_text_1 = text_newyork[1]
newyork_text = ny_text_0 + ny_text_1

text_cali= []
for pnum in pages:
    page = cali_law.getPage(pnum)
    text_cali.append(page.extractText())
cali_text_0 = text_cali[0]
cali_text_1 = text_cali[1]
cali_text = cali_text_0 + cali_text_1

variables = [alabama_text , florida_text , newyork_text , cali_text]

#defining a function to clean the text, by lowering all the words, checking language. 
for t in variables:
    def lower(t):
        spell = Speller(lang='en')
        texts = spell(t)
        return ' '.join([w.lower() for w in word_tokenize(t)])

#assigning variables for all lowerecase texts for each state
alabama_lower = lower(alabama_text)
florida_lower = lower(florida_text)
newyork_lower = lower(newyork_text)
cali_lower = lower(cali_text)
lower_case = [alabama_lower , florida_lower , newyork_lower , cali_lower]

#creating trigrams for the text from each state's law text
for l in lower_case:
    def tokenized(l):
        token = l.split()
        esBigrams = ngrams(token, 3)
        return list(esBigrams)[:1000]
        
#creating tokens for each of the states' trigrams 
alabama_tokens = tokenized(alabama_lower)
florida_tokens = tokenized(florida_lower)
newyork_tokens = tokenized(newyork_lower)
cali_tokens = tokenized(cali_lower)
tokens = [alabama_tokens, florida_tokens , newyork_tokens, cali_tokens]

#creating a dataframe for each state containing the trigrams 
for token in tokens:
    def df_state(token):
        df_state = pd.DataFrame(token)
        df_state['words'] = df_state[df_state.columns[0:]].apply(lambda x: ','.join(x.dropna().astype(str)), axis=1)
        df_state = df_state.replace(',',' ', regex=True)
        return df_state

df_alabama = df_state(alabama_tokens)
#df_alabama
df_florida = df_state(florida_tokens)
#df_florida
df_newyork = df_state(newyork_tokens)
#df_newyork
df_cali = df_state(cali_tokens)
#df_cali

#NLP for harshness

#creating sentiment scale for harsh and easy laws
harsh_laws = ['prohibited from possessing' , 'from possessing firearms' , 'authorized to order' , 'necessary to protect' , 'requiring surrender .' , 'including dating partners' , 'relinquish their firearms' , 'background checks are' , 'must confiscate firearms']
easy_laws = ['prohibition does not' , 'does not apply' , 'does not prohibit'  , 'are not statutorily' , 'may include prohibiting' , 'buyer is not']

#Measuring Harshness for Alabama
harsh_count_alabama  = 0
easy_count_alabama = 0 
for t in df_alabama['words']:
    if t in harsh_laws:
        harsh_count_alabama += 1
    elif t in easy_laws:
        easy_count_alabama += 1

print('Harsh:', harsh_count_alabama , '\nDown:', easy_count_alabama )
harshness_alabama = harsh_count_alabama - easy_count_alabama
harshness_alabama
 
##Measuring Harshness for Florida
harsh_count_florida  = 0
easy_count_florida= 0 
for t in df_florida['words']:
    if t in harsh_laws:
        harsh_count_florida += 1
    elif t in easy_laws:
        easy_count_florida += 1
print('Harsh:', harsh_count_florida , '\nDown:', easy_count_florida )
harshness_florida = harsh_count_florida - easy_count_florida
harshness_florida
        
##Measuring Harshness for Newyork
harsh_count_newyork  = 0
easy_count_newyork= 0 
for t in df_newyork['words']:
    if t in harsh_laws:
        harsh_count_newyork += 1
    elif t in easy_laws:
        easy_count_newyork += 1
print('Harsh:', harsh_count_newyork , '\nDown:', easy_count_newyork )
harshness_newyork= harsh_count_newyork - easy_count_newyork
harshness_newyork

##Measuring Harshness for California
harsh_count_cali  = 0
easy_count_cali = 0 
for t in df_cali['words']:
    if t in harsh_laws:
        harsh_count_cali += 1
    elif t in easy_laws:
        easy_count_cali += 1
print('Harsh:', harsh_count_cali , '\nDown:', easy_count_cali )
harshness_cali= harsh_count_cali - easy_count_cali
harshness_cali        


#Creating a dataframe to represent harshness for  all the four states
harsh_count = [harsh_count_alabama , harsh_count_florida , harsh_count_newyork , harsh_count_cali]
easy_count = [easy_count_alabama , easy_count_florida , easy_count_newyork , easy_count_cali]
harshness = [harshness_alabama , harshness_florida , harshness_newyork, harshness_cali]
states = ['Alabama' , 'Florida' , 'Newyork' , 'California']
df_harshness = pd.DataFrame(states)
df_harshness['harsh_count'] = harsh_count
df_harshness = df_harshness.rename(columns={0:'State'})
df_harshness['easy_count'] = easy_count
df_harshness['harshness'] = harshness
df_harshness.to_csv("harshness_law.csv")

#creating plot to summarize harshness of the states 
fig, ax1 = plt.subplots()
ax1.bar(df_harshness['State'] ,df_harshness['harshness'] , label='Level of Harshness' , color = 'maroon')
ax1.set_title('Firearm Protection Law: Level of Harshness')
ax1.legend()
ax1.autoscale_view()
plt.show()
fig.savefig('ax1_harshness.png')



