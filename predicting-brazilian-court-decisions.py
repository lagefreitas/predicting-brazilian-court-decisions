#!/usr/bin/env python
# coding: utf-8

# # Predicting Brazilian Court Decisions
# 
# This source code was used at the paper entitled _Predicting Brazilian Court Decisions_ submitted to PeerJ Computer Science journal under ID `CS-2020:08:51609`. 
# 
# This paper was first published as a Technical Report on April 20th, 2019 at arXiv: 
# 
# * _André Lage-Freitas, Héctor Allende-Cid, Orivaldo Santana, Lívia Oliveira-Lage. "Predicting Brazilian Court Decisions". Technical Report. 2019. http://arxiv.org/abs/1905.10348_.
# 
# We used this Python program for Steps 4, 5, 6, 7, 8, and 9 of our methodology. The Web scraper, regarding the Steps 1, 2, and 3 of our methodology, are under JusPredict (https://www.juspredict.com.br) Intellectual Property.

# In[1]:


import requests
import time
from datetime import datetime
import os.path
from os import listdir
from os.path import isfile, join
import re
from bs4 import BeautifulSoup
import textwrap
import logging
import pandas as pd
import numpy as np
import unidecode
import statistics
import seaborn as sns
import string
import time
import random
import copy
import nltk
from nltk.stem.porter import *
nltk.download('rslp')
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Create directories for storing screaped data
now = datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H:%M:%S")
if not os.path.exists(timestamp):
    os.makedirs(timestamp) # BUGFIX remove automatically created file named 'Icon?' in OSX operating systems
    print('Directory created:' + timestamp)
    
# Logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logfile = timestamp + '/' + timestamp + '.log'
logging.basicConfig(filename=logfile,level=logging.DEBUG)
print('logfile is:',logfile)
logging.info('logfile is: ' + logfile)

print('\n[DEBUG] Log ready for you tail it (PRESS ANY KEY TO CONTINUE):\ntail -f ' + logfile + '\n')
# input()


# # STEP 4
# 
# # Data loading and cleaning

# In[145]:


FILE_DECISION_CSV = 'dataset.csv'

# Loading the labeled decisions
data = pd.read_csv(FILE_DECISION_CSV, sep='<=>',header=0)
print('data.shape=' + str(data.shape) + ' full data set')
# Removing NA values
data = data.dropna(subset=[data.columns[9]])# decision_description
data = data.dropna(subset=[data.columns[11]])# decision_label
print('data.shape=' + str(data.shape) + ' dropna')
# Removing duplicated samples
# df = data.groupby(['process_number']).size().reset_index(name='count')
# print('df.shape=' + str(df.shape) + ' removed duplicated samples by process number')
data = data.drop_duplicates(subset=[data.columns[1]]) # process_number
print('data.shape=' + str(data.shape) + ' removed duplicated samples by process_number')
data = data.drop_duplicates(subset=[data.columns[9]]) # decision_description
print('data.shape=' + str(data.shape) + ' removed duplicated samples by decision_description')
# Removing not relevant decision labels and decision not properly labeled
data = data.query('decision_label != "conflito-competencia"')
print('data.shape=' + str(data.shape) + ' removed decisions labeled as conflito-competencia')
data = data.query('decision_label != "prejudicada"')
print('data.shape=' + str(data.shape) + ' removed decisions labeled as prejudicada')
data = data.query('decision_label != "not-cognized"')
print('data.shape=' + str(data.shape) + ' removed decisions labeled as not-cognized')
data_no = data.query('decision_label == "no"')
print('data_no.shape=' + str(data_no.shape))
data_yes = data.query('decision_label == "yes"')
print('data_yes.shape=' + str(data_yes.shape))
data_partial = data.query('decision_label == "partial"')
print('data_partial.shape=' + str(data_partial.shape))
# Merging decisions whose labels are yes, no, and partial to build the final data set
data_merged = data_no.merge(data_yes, how='outer')
data = data_merged.merge(data_partial, how='outer')
print('data.shape=' + str(data.shape) + ' merged decisions whose labels are yes, no, and partial')
# Removing decision_description and decision_labels whose values are -1 and -2
indexNames = data[ (data['decision_description'] == str(-1)) |                    (data['decision_description'] == str(-2)) |                    (data['decision_label'] == str(-1)) |                    (data['decision_label'] == str(-2)) ].index
# print('indexNames='+str(len(indexNames)))
data.drop(indexNames, inplace=True)
print('data.shape=' + str(data.shape) + ' removed -1 and -2 decision descriptions and labels')


# ### Text pre-processing 
# 
# Operates over $X$, i.e., `data['decision_description']`.
# 
# * Lower case
# * Stemmer
# * Removes dots
# * Removes stop-words
# * Removes accents (normalize)

# In[146]:


import unidecode

#Stemming leaves only the root of the word. 
stemmer = PorterStemmer()
#Create set of stopwords
stopwords = nltk.corpus.stopwords.words('portuguese')

start = time.time()
index = 0
# data_decision_description_preprocessed=[]

data['decision_description'] = data['decision_description'].str.lower()
data['decision_description'] = data['decision_description'].str.replace('.','')
# i = 0    
for i in range(len(data['decision_description'])):   
    data['decision_description'][i] = unidecode.unidecode(data['decision_description'][i])
    lineanueva=''
    for pal in data['decision_description'][i].split():
        if pal not in stopwords:
            lineanueva=lineanueva+stemmer.stem(pal)+" "
    data['decision_description'][i]=lineanueva
    #i += 1

end = time.time()
total_time = end - start
print('Execution time in seconds: ' + str(total_time) )
print('Execution time in minutes: ' + str(total_time/60) )
data.head()


# ### Data set statistics

# In[147]:


data['totalwords']=data['decision_description'].str.count(' ') + 1
data['totalwords'].describe()


# In[148]:


cjto = set()
for index,row in data.iterrows():
    for word in row['decision_description'].split():
        cjto.add(word)
print(len(cjto))


# # STEPS 5 and 6
# 
# ### Text Segmentation and Decision Classification (Labeler)

# In[144]:


us = []
us_notfound = []
us_text =[]
us_notfound_text = []
count_decisions_labeled = 0
count_decisions_NOT_labeled = 0

class Ementa:
    ementa_filepath = ''
    process_number = '' 
    process_type = '' 
    orgao_julgador = ''
    publish_date = ''
    judgment_date = ''
    judge_relator = ''
    ementa_text = '' # the whole Ementa text
    decisions = []
    decision = {
        'description': '-1', # the text that contains all the decision description but the decision itself, e.g.: RECURSO DE APELAÇÃO. MUNICÍPIO DE...
        'text': '-1', # the sentece/text that contains the decision, e.g.: RECURSO CONHECIDO E PROVIDO
        'label': '-1', # the decision classification, ex.: yes, no, partial, etc. 
        'unanimity_text': '-1', # the text that has information about the unanimity, e.g.: UNANIMIDADE        
        'unanimity': '-1' # the unanimity classification: unanimity or not-unanimity. Automatically filled by the system.
    }

    def __init__(self, ementa_filepath):
        logline = '__init__: '        
        self.ementa_filepath = ementa_filepath
        
        string = self.get_metadata('esajLinkLogin downloadEmenta')
        string = self.cleanhtml(str(string))
        string = string.split(' ')
        found_process_number = False
        logging.debug( str(logline) + 'Found a string for process number and type: ' + str(string) + '. File: ' + str(self.ementa_filepath) ) 
        if len(string) < 2:
            logging.error( str(logline) + 'Not found process number and type (len is zero): ' + str(string) + '. File: ' + str(self.ementa_filepath) ) 
        else: 
            self.process_type = string[0]
            self.process_number = string[1]
            found_process_number = True
            logging.debug('Found process number/type: ' + str(self.process_number) + '/' + str(self.process_type) + '. File: ' + str(self.ementa_filepath) )
                
        if not found_process_number:
            logging.error( str(logline) + 'Not found process number and type: ' + str(self.process_number) + '. File: ' + str(self.ementa_filepath) ) 

        self.orgao_julgador = self.get_metadata('Órgão julgador')
        self.publish_date = self.get_metadata('Data de publicação')
        self.judgment_date = self.get_metadata('Data do julgamento')
        self.judge_relator = self.get_metadata('Relator')
        self.ementa_text = self.get_ementa()
        self.decisions = self.decision_labeling()
        
    def get_metadata(self, metadata):
        logline = 'get_metadata: '
        logging.debug( str(logline) + 'Called for file path: ' + str(self.ementa_filepath) )
        file = open(self.ementa_filepath, mode='r',encoding='utf-8')
        soup = BeautifulSoup(file, 'html.parser')
        strings = soup.find_all(string=True)
        for s in strings:
            if str(metadata) in s:
                index = strings.index(s)
                try:
                    logging.debug( str(logline) + 'Found Ementa metadata "' + str(metadata) + '" (' + str(strings[index + 1]) + ')  in file: ' + str(self.ementa_filepath) ) 
                    file.close()
                    return str(strings[index + 1])
                except:
                    logging.error( str(logline) + 'Not found Ementa metadata "' + str(metadata) + '" in file: ' + str(self.ementa_filepath) )
        file.close()
        return '-1'    
        
    def cleanhtml(self, raw_html):
      cleanr = re.compile('<.*?>')
      cleantext = re.sub(cleanr, '', raw_html)
      return cleantext

    # Step 6: Decision Classification (Labeler)
    def decision_labeling(self):
        global count_decisions_labeled
        global count_decisions_NOT_labeled        
        logline = 'decision_labeling: '
        
        logging.debug( str(logline) + 'Before removing symbols:  ' + str(self.ementa_text) )        
        sentences = self.remove_symbols( self.ementa_text ).split('.') # sentences = self.ementa_text.split('.')
        logging.debug( str(logline) + '\nAfter removing symbols:  ' + str(sentences) )
        
        decision_descriptions = ['']
        decisions = ['']
        decision_id = 0
        decision_labels = ['']
        label_unanimity = ''

        u = '-1'
        count = 1    
        
        # Step 5: Text Segmentation
        def lookfor_decision():
            global count_decisions_labeled            
            local_logline = logline + 'lookfor_decision: '
            logging.debug( str(local_logline) + 'Looking for decision and labeling it...' )            
            decision_fonud_locally = False
            count = 1
            for s in reversed(sentences):
            # Look for decisions up to the 2 last sentences, the first s usually is '' or ' '
                if count > 3:
                    break
                d = self.get_decision_label(s)
                if d != '-1' and d != '-2':
                    self.decision["label"] = d
                    self.decision["text"] = s
                    logging.info( str(local_logline) + 'Decision label is "' + str(d) + '" > ' + str(s) )
                    count_decisions_labeled += 1
                    decision_fonud_locally = True
                    del sentences[ sentences.index(s) ] # remove decision text so decision description won't have the decision itself
                    self.decision["description"] = '.'.join( sentences )
    #                 break
                    return True
                count += 1
            # full search for "conflito de competencia"
            if not decision_fonud_locally:
                for s in sentences:
                    if self.is_keyword(self.get_conflito_competencia(), s):
                        d = 'conflito-competencia'
                        self.decision["label"] = d
                        self.decision["text"] = s
                        count_decisions_labeled += 1
                        logging.info( str(local_logline) + 'Decision label is "' + str(d) + '" > ' + str(s) + ' File: ' + str(self.ementa_filepath) )
    #                     found_decision = True
                        self.decision["description"] = '.'.join( sentences )        
    #                     break
                        return True
  
        # Step 5: Text Segmentation
        logging.debug( str(logline) + 'Looking for information on unanimity...' )
        for s in reversed(sentences):
            # Look for unanimity info up to the 2 last sentences, the first s is '' or ' '
            if count > 3:
                break
            u = self.get_unanimity_label(s)
            if u != '-1' and s != ' ' and s != '':
                self.decision["unanimity"] = u
                self.decision["unanimity_text"] = s
                logging.info( str(logline) + 'Unanimity is "' + str(u) + '" > ' + str(s) )
                us.append(u)
                us_text.append(s)
                # if the sentence only contains unanimity contents, then remove it to make it easier to get the decisions afterwards
                if len(s) < 35: 
                    logging.info( str(logline) + 'This unanimity sentence will be deleted from sentences list: ' + str(s) + ' > File: ' + str(self.ementa_filepath) )
                    del sentences[ sentences.index(s) ] 
                break
            count += 1
        if u == '-1':
            self.decision["unanimity"] = '-2'
            self.decision["unanimity_text"] = '-2'            
            logging.error( str(logline) + 'Unanimity info NOT found "' + str(u) + '" > ' + str(s) )
            us_notfound.append(u)
            us_notfound_text.append(s)  
        
        # Step 5: Text Segmentation
        # Step 6: Decision Classification (Labeler)
        logging.debug( str(logline) + 'Looking for decision and labeling it...' )
        decision_found = lookfor_decision()
        if not decision_found:
            logging.debug( str(logline) + 'Removing upper case sentences from Ementa and REPEAT: Looking for decision and labeling it...' )
            for s in reversed(sentences):
                for c in s:
                    if c.islower():# and not re.search('\d', s): 
#                         logging.debug( str(logline) +  )
                        del sentences[ sentences.index(s) ]
                        break
            decision_found = lookfor_decision()
            
        if not decision_found:            
            self.decision["label"] = '-2'
            self.decision["text"] = '-2'
            count_decisions_NOT_labeled += 1
            logging.error( str(logline) + 'Decision NOT found ' + str(s) + ' File: ' + str(self.ementa_filepath) )

        if self.decision['label'] == '-2' or self.decision['label'] == '-1':
            logging.debug( str(logline) + 'FILE    >>> ' + str(self.ementa_filepath) )        
    
    # Basic data cleaning
    def remove_symbols(self, string):
        symbols = ['º','°',]        
        abbreviations = [ ['art.'  ,  'n.'  , 'inc.' ,    'cp.'     ,    'cp '       ,      'cp,'    ,          'cpc'           ] ,                          ['ARTIGO','NUMERO','INCISO','CODIGO PENAL.','CODIGO PENAL ','CODIGO PENAL,','CODIGO DE PROCESSO CIVIL'] ]
        for index in range(len(abbreviations[0])):
            abbr = abbreviations[0][index]
            replacement = abbreviations[1][index]
            logging.debug('Replacing all abbreviations "' + str(abbr) + '" by "' + str(replacement) + '"')
            if abbr in unidecode.unidecode(string).lower():
                exp = re.compile(re.escape(abbr), re.IGNORECASE)
                string = exp.sub(replacement, string)
        string = re.sub('(?<=\d)\.(?=\d)', '', string)
        logging.debug('All points between numbers were removed: "' + str(string) + '"')        
        string = re.sub('§', 'PARAGRAFO', string)
        logging.debug('All "§" symbols were replaced by "PARAGRAFO"' + str(string) + '"')        
        for s in symbols:
            logging.debug('Removing all symbols "' + str(s) + '"')
            string = re.sub(s, '', string)
        return string        
        
    # Step 6: Decision Classification (Labeler)
    def get_decision_label(self, s):
        logline = 'get_decision_label: '        
        if len(s) == 0 or s == ' ':
            logging.debug( str(logline) + 'Could NOT get Decision: len(s)="' + str(len(s)) + '", s="' + str(s) + '"')
            return '-1'
        else:
            not_a_decision = ['comprov', 'concurso publico', ' que ', ' que,']
            no = ['nao defir','nao defer','neg','desprov','indef','nao acolh',                  'inacolh','rejeit','improv','nao prov','improcedente o pedido','improcedente',                 'confirmar a sentenca','ratificar integralmente a sentenca',                 'ratificar sentenca','ratificar a sentenca','sentenca confirmada',                  'sentenca mantida','sentenca de primeiro grau mantida',                  'manutencao da concessao da seguranca']
            yes = ['a decisao agravada deve ser reformada','anulacao da sentenca',                   'anular sentenca','reformar a sentenca','reformar sentenca',                   'de oficio, reconhecer','de oficio reconhecer','acolhido','acolhimento',                   'defir','defer','prove','dou provimento','provimento','conced',                   'acolh','tese aceita','ordem concedida','pedido concedido','provido','provida',                  'sentenca anulada','decisao reformada','sentenca reformada' ,'acolhimento',                  'reforma da decisao']
            partial = ['parcial','em parte']
            not_cognized = ['recurso nao conhec','negou conhec','negou apelo',                            'apelo nao conhecido','nao prover conhecimento','conhecimento nao provido',                            'nao proveu conhecimento','nao provido conhecimento','nao provido apelo',                            'apelo nao conhecido','apelo nao provido','nao reconhecer conhecimento',                            'conhecimento nao conhecido', 'nao conhecimento']
                            #TODO BUGFIX-REGEX use regex for these terms
#                             'nao reconhec* conhec*','nao prov* conhec*',\
#                             'nao prov* apelo',,]
            prejudicada = ['prejudic']
            if self.is_keyword(no, s):
                return 'no'
            if self.is_keyword(partial, s):
                return 'partial'
            if self.is_keyword(prejudicada, s):
                return 'prejudicada'
            if self.is_keyword(yes, s):
                return 'yes'
            if self.is_keyword(not_cognized, s):
                return 'not-cognized'
            if self.is_keyword(self.get_conflito_competencia(), s):
                return 'conflito-competencia'
            return '-2'

    # Step 6: Decision Classification (Labeler)
    def get_conflito_competencia(self):
        conflito_competencia = ['conflito de competencia','conflito negativo de competencia',
                                'CONFLITO NEGATIVO DE COMPETÊNCIA']
        return conflito_competencia

    # Checks whether a keyword is in a list of strings by ignoring case sensitive and convertging Unicode characters.
    def is_keyword(self, keyword_list, s):
        logline = 'is_keyword: '
        for keyword in keyword_list:
            logging.debug( str(logline) + 'Checking if keyword=' + str(keyword) + ' is in ' + str(s) )
            if keyword in unidecode.unidecode(s).lower():
                return True
        return False        
    
    # Step 6: Decision Classification (Labeler)
    def get_unanimity_label(self, s):
        if 'unanim' in s.lower() or 'unânim' in s.lower():
            return 'unanimity'
        elif 'por maioria' in s.lower() or 'TODO' in s.lower():            
            return 'not-unanimity'
        else:
            return '-1'
    
    # Loads Ementa text
    def get_ementa(self):
        logline = 'get_ementa: '
        logging.debug( str(logline) + 'Called for file path: ' + str(self.ementa_filepath) )
        file = open(self.ementa_filepath, mode='r',encoding='utf-8')
        soup = BeautifulSoup(file, 'html.parser')
        e = soup.find_all(name='p')#, string=re.compile('prov', re.IGNORECASE), string=True, recursive=True)
        if len(e) <= 1:
            logging.error( str(logline) + 'Could NOT find Ementa in file ' + str(self.ementa_filepath))
            file.close()            
            return '-1'
        else:
            del e[0]
            e2 =[str(i) for i in e]
            ementa_text = ' '.join(e2) 
            ementa_text = re.sub(r'\x96', '', ementa_text)
            ementa_text = re.sub(r'  ', ' ', ementa_text)
            ementa_text = self.cleanhtml(ementa_text)
            logging.debug( str(logline) + '\nFound this Ementa in file ' + str(self.ementa_filepath) + ': ' + str(ementa_text))            
        file.close()            
        return ementa_text  

    
#
#  LAUNCH
# 
downloaded_ementas = [f for f in listdir(workingdir_ementas) if isfile(join(workingdir_ementas, f))]
downloaded_ementas = [workingdir_ementas + s for s in downloaded_ementas]
downloaded_ementas = sorted(downloaded_ementas)


ementas = []
counter = 0
FILE_DECISION_CSV = 'dataset.csv'
decisions_csv = open(FILE_DECISION_CSV,'w')
csv_header = 'ementa_filepath<=>process_number<=>process_type<=>orgao_julgador<=>publish_date<=>judgment_date<=>judge_relator<=>ementa_text<=>decisions<=>decision_description<=>decision_text<=>decision_label<=>decision_unanimity_text<=>decision_unanimity\n'
decisions_csv.write( str(csv_header) )

logging.root.setLevel(logging.INFO)

datum = []
start = time.time()
for f in downloaded_ementas:
#     if len(datum) > 10: break
    e = Ementa(f)
    ementas.append(e)
    if not e.decision['label'] == '-2' and not e.decision['label'] == '-1':
        data = str(e.ementa_filepath) + '<=>'  +            str(e.process_number) + '<=>'  +            str(e.process_type) + '<=>'  +            str(e.orgao_julgador) + '<=>'  +            str(e.publish_date) + '<=>'  +            str(e.judgment_date) + '<=>'  +            str(e.judge_relator) + '<=>'  +            str(e.ementa_text) + '<=>'  +            str(e.decisions) + '<=>'  +            str(e.decision['description']) + '<=>'  +            str(e.decision['text']) + '<=>'  +            str(e.decision['label']) + '<=>'  +            str(e.decision['unanimity_text']) + '<=>'  +            str(e.decision['unanimity']) + '\n'
        decisions_csv.write( str(data) )
        datum.append(data)
decisions_csv.close()

end = time.time()
total_time = end - start

logging.debug('count_decisions_labeled=' + str(count_decisions_labeled) + '  count_decisions_NOT_labeled=' + str(count_decisions_NOT_labeled) + '  in ' + str(total_time)  + ' secs')


# # STEPS 7, 8, and 9

# ## Scenario 1: data set

# In[6]:


print('\n== SCENARIO 1 - Data set ==\n')
# SCENARIO 1
data_scenario1 = data
print('data_scenario1.shape=' + str(data_scenario1.shape))
# Loading data into vectors and print the frequency they appear
X_scenario1=data_scenario1['decision_description']
y_scenario1=data_scenario1['decision_label']
unique, counts = np.unique(y_scenario1, return_counts=True)
dict_scenario1 = dict(zip(unique, counts))
print('Decision label frequency: ' + str(dict_scenario1) )


# ## Scenarios 2 and 3: data sets
# 
# Balancing data set (distributes data set uniformly).
# 
# Makes the number of `no` decisions the same as `partial` as there are a lot more `no` decisions than `partial` and `yes`.

# In[8]:


# As there are much more `no`-labled decisions, we randomly remove `no` decisions to have the same number of `partial` decisions (partial is greater then yes in this data set)
data_no = data[ data.decision_label == 'no' ]
print('data_no.shape=' + str(data_no.shape) + ' only no-labeled decisions')
data_yes = data[ data.decision_label == 'yes' ]
print('data_yes.shape=' + str(data_yes.shape) + ' only yes-labeled decisions')
data_partial = data[ data.decision_label == 'partial' ]
print('data_partial.shape=' + str(data_partial.shape) + ' only partial-labeled decisions')

y_dict=data['decision_label']
unique, counts = np.unique(y_dict, return_counts=True)
dict_ = dict(zip(unique, counts))
rows = np.random.choice(data_no.index.values, dict_['partial'])
data_no = data_no.loc[rows]
print('data_no.shape=' + str(data_no.shape) + ' only no-labeled decisions')

print('\n== SCENARIO 2 - Data set ==\n')
# SCENARIO 2
data_scenario2 = data_no.merge(data_yes, how='outer')
data_scenario2 = data_scenario2.merge(data_partial, how='outer')
print('data_scenario2.shape=' + str(data_scenario2.shape))
# Loading data into vectors and print the frequency they appear
X_scenario2=data_scenario2['decision_description']
y_scenario2=data_scenario2['decision_label']
unique, counts = np.unique(y_scenario2, return_counts=True)
dict_scenario2 = dict(zip(unique, counts))
print('Decision label frequency: ' + str(dict_scenario2) )

print('\n== SCENARIO 3 - Data set ==\n')
# SCENARIO 3
data_scenario3 = data_scenario1
data_scenario3 = data_scenario3.replace(to_replace='partial',value='yes')
print('data_scenario3.shape=' + str(data_scenario3.shape))
# Loading data into vectors and print the frequency they appear
X_scenario3=data_scenario3['decision_description']
y_scenario3=data_scenario3['decision_label']
unique, counts = np.unique(y_scenario3, return_counts=True)
dict_scenario3 = dict(zip(unique, counts))
print('Decision label frequency: ' + str(dict_scenario3) )


# ### Using tf-idf to text representation
# 
# ### For $X$

# In[9]:


vectorizer = TfidfVectorizer()
X_tfidf_scenario1, X_tfidf_scenario2, X_tfidf_scenario3 = '','',''

vectorizer.fit(data_scenario1['decision_description'])
vectorizer.fit(data_scenario2['decision_description'])
vectorizer.fit(data_scenario3['decision_description'])
X_tfidf_scenario1 = vectorizer.transform(data_scenario1['decision_description'])
X_tfidf_scenario2 = vectorizer.transform(data_scenario2['decision_description'])
X_tfidf_scenario3 = vectorizer.transform(data_scenario3['decision_description'])

indices = np.argsort(vectorizer.idf_)[::-1]
print('indices = ' + str(indices))


# ### For $y$

# In[10]:


le = preprocessing.LabelEncoder()

y_coded_scenario1=le.fit_transform(y_scenario1)
print(list(le.inverse_transform([0, 1, 2])))
y_coded_scenario2=le.fit_transform(y_scenario2)
print(list(le.inverse_transform([0, 1, 2])))
y_coded_scenario3=le.fit_transform(y_scenario3)
print(list(le.inverse_transform([0, 1])))

y_coded_scenario3


# ## Configurations: Parallel execution and Cross Validation

# In[11]:


# parallel processing configuration for hyperparameter search
nof_threads = 40 # GranColoso has 40 cores


# In[12]:


# Cross Validation number of folds
cv_nfolds = 5


# ## Model training

# ### Defining functions for Choosing hyperparameters

# In[13]:


def svc_param_selection(X, y, nfolds):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    degree=[1,2,3,4,5,6,7,8,9]
    kernel=[ 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=nfolds, n_jobs=nof_threads)
    grid_search.fit(X, y)
    grid_search.best_estimator_
    return grid_search.best_estimator_ 

# learning rate=[0.1, 0.2, 0.3], maximum depth from 1 to 6, minimum child weight from 1 to 5, and number of estimators from 100 to 1000 in steps of 100}. 
def xgboost_param_selection(X, y, nfolds):
    print('xgboost_param_selection: ' + 'param_grid')
    param_grid = {'objective':['multi:softmax'],
              'learning_rate': [0.1, 0.2, 0.3], #so called `eta` value
              'max_depth': [1,2,3,4,5,6],
              'min_child_weight': [1,2,3,4,5],
              'silent': [1],
              'subsample': [1],
              'colsample_bytree': [1],
              'n_estimators': [1000], #number of trees, change it to 1000 for better results
              'missing':[-999]}
    print('xgboost_param_selection: ' + 'GridSearchCV')    
    grid_search = GridSearchCV(XGBClassifier(), param_grid, cv=nfolds, n_jobs=nof_threads)
    print('xgboost_param_selection: ' + 'grid_search.fit')        
    grid_search.fit(X, y)
    print('xgboost_param_selection: ' + 'grid_search.best_estimator_')        
    grid_search.best_estimator_
    return grid_search.best_estimator_ 

# \emph{minimum number of records from 10 to 100 in steps of 10, max depth of the tree from 1 to 20 in steps of 2}. 
def dt_param_selection(X, y, nfolds):
    param_grid = {'min_samples_split' : range(10,100,10),'max_depth': range(1,20,2)}
    grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=nfolds, n_jobs=nof_threads)
    grid_search.fit(X, y)
    grid_search.best_estimator_
    return grid_search.best_estimator_

# The parameters of Random Forest were the following: \emph{number of estimators from 100 to 1000 in steps of 100 and max depth from 1 to 6}.
def rf_param_selection(X, y, nfolds):
    param_grid = {'max_depth': range(1,6), 'n_estimators': range(100,1000,100)}
    grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=nfolds, n_jobs=nof_threads)
    grid_search.fit(X, y)
    grid_search.best_estimator_
    return grid_search.best_estimator_


# ### Remark
# 
# 
# # Scenario 1
# 
# ### Hyperparameter tuning: Scenario 1

# In[14]:


start = time.time()

# Training models
cv = ShuffleSplit(n_splits=cv_nfolds, test_size=0.2, random_state=1)

print('Choosing SVM hyperparameters...')
best_SVM_scenario1 = svc_param_selection(X_tfidf_scenario1, y_coded_scenario1, cv_nfolds)
print('best_SVM_scenario1' + str(best_SVM_scenario1))

print('Choosing XGBoost hyperparameters...')
best_XG_scenario1 = xgboost_param_selection(X_tfidf_scenario1, y_coded_scenario1, cv_nfolds)
print(best_XG_scenario1)

print('Choosing Decision Tree hyperparameters...')
best_DT_scenario1 = dt_param_selection(X_tfidf_scenario1, y_coded_scenario1, cv_nfolds)
print(best_DT_scenario1)

print('Choosing Random Forest hyperparameters...')
best_RF_scenario1 = rf_param_selection(X_tfidf_scenario1, y_coded_scenario1, cv_nfolds)
print(best_RF_scenario1)


# Loading models
clf_GaussianNB_scenario1 = GaussianNB()
clf_DT_scenario1 = best_DT_scenario1
clf_SVM_scenario1 = best_SVM_scenario1
#clf_RF_scenario1 = RandomForestClassifier(n_estimators=100, max_depth=4,random_state=0)
clf_RF_scenario1 = best_RF_scenario1
# clf5=XGBClassifier(learning_rate=0.1, n_estimators=190, max_depth=5, min_child_weight=2, objective="multi:softmax", subsample=0.9, colsample_bytree=0.8, seed=23333)
clf_XGB_scenario1 = best_XG_scenario1

end = time.time()
total_time = end - start
print('Execution time in seconds: ' + str(total_time) )
print('Execution time in minutes: ' + str(total_time/60) )


# ### Model training: Scenario 1

# In[15]:


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
kf = StratifiedKFold(n_splits=cv_nfolds ) 

# precision
precision_GNB=[] # list of precision results: precision_GNB
scores_clf_DT_scenario2=[] # list of precision results: precision_DT
scores_clf_SVM_scenario2=[]
scores_clf_RF_scenario2=[]
scores_clf_XGB_scenario2=[]

# recall
scores_clf_GNB_scenario2_2=[]
scores_clf_DT_scenario2_2=[]
scores_clf_SVM_scenario2_2=[]
scores_clf_RF_scenario2_2=[]
scores_clf_XGB_scenario2_2=[]

# accuracy
scores_clf_GNB_scenario2_3=[]
scores_clf_DT_scenario2_3=[]
scores_clf_SVM_scenario2_3=[]
scores_clf_RF_scenario2_3=[]
scores_clf_XGB_scenario2_3=[]

# f1
scores_clf_GNB_scenario2_4=[]
scores_clf_DT_scenario2_4=[]
scores_clf_SVM_scenario2_4=[]
scores_clf_RF_scenario2_4=[]
scores_clf_XGB_scenario2_4=[]

# F1-socre
# It is calculated by using precision and recall

start = time.time()

for train_index, test_index in kf.split(X_tfidf_scenario1,y_coded_scenario1):
    X_tfidf_scenario1.toarray(), y_coded_scenario1
    X_train, X_test = X_tfidf_scenario1[train_index], X_tfidf_scenario1[test_index]
    y_train, y_test = y_coded_scenario1[train_index], y_coded_scenario1[test_index]

    clf_GaussianNB_scenario1.fit(X_train.toarray(),y_train)
    clf_DT_scenario1.fit(X_train.toarray(),y_train)
    clf_RF_scenario1.fit(X_train.toarray(),y_train)
    clf_SVM_scenario1.fit(X_train,y_train)
    clf_XGB_scenario1.fit(X_train.toarray(),y_train)
    
    y_pred_GNB=clf_GaussianNB_scenario1.predict(X_test.toarray())
    y_pred_DT=clf_DT_scenario1.predict(X_test)
    y_pred_SVM=clf_SVM_scenario1.predict(X_test)
    y_pred_RF=clf_RF_scenario1.predict(X_test)
    y_pred_XGB=clf_XGB_scenario1.predict(X_test)

    # scores_clf_GNB_scenario2 = list to store the results of Scenario 1
    precision_GNB.append(precision_score(y_test, y_pred_GNB, average='macro'))
    scores_clf_DT_scenario2.append(precision_score(y_test, y_pred_DT, average='macro'))
    scores_clf_SVM_scenario2.append(precision_score(y_test, y_pred_SVM, average='macro'))
    scores_clf_RF_scenario2.append(precision_score(y_test, y_pred_RF, average='macro'))
    scores_clf_XGB_scenario2.append(precision_score(y_test, y_pred_XGB, average='macro'))
    
    scores_clf_GNB_scenario2_2.append(recall_score(y_test, y_pred_GNB, average='macro'))
    scores_clf_DT_scenario2_2.append(recall_score(y_test, y_pred_DT, average='macro'))
    scores_clf_SVM_scenario2_2.append(recall_score(y_test, y_pred_SVM, average='macro'))
    scores_clf_RF_scenario2_2.append(recall_score(y_test, y_pred_RF, average='macro'))
    scores_clf_XGB_scenario2_2.append(recall_score(y_test, y_pred_XGB, average='macro'))
    
    scores_clf_GNB_scenario2_3.append(accuracy_score(y_test, y_pred_GNB))
    scores_clf_DT_scenario2_3.append(accuracy_score(y_test, y_pred_DT))
    scores_clf_SVM_scenario2_3.append(accuracy_score(y_test, y_pred_SVM))
    scores_clf_RF_scenario2_3.append(accuracy_score(y_test, y_pred_RF))
    scores_clf_XGB_scenario2_3.append(accuracy_score(y_test, y_pred_XGB))
    
    scores_clf_GNB_scenario2_4.append(f1_score(y_test, y_pred_GNB, average='macro'))
    scores_clf_DT_scenario2_4.append(f1_score(y_test, y_pred_DT, average='macro'))
    scores_clf_SVM_scenario2_4.append(f1_score(y_test, y_pred_SVM, average='macro'))
    scores_clf_RF_scenario2_4.append(f1_score(y_test, y_pred_RF, average='macro'))
    scores_clf_XGB_scenario2_4.append(f1_score(y_test, y_pred_XGB, average='macro'))


end = time.time()
total_time = end - start
print('Execution time in seconds: ' + str(total_time) )
print('Execution time in minutes: ' + str(total_time/60) )


# ### Confusion Matrix: Scenario 1

# In[17]:


from sklearn.metrics import plot_confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
class_names = np.unique(y_test)

# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ('Scenario 1: XGBoost model\nNormalized Confusion Matrix', 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(clf_XGB_scenario1, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize,
                                 xticks_rotation='horizontal')
    disp.ax_.set_title(title)#, fontsize=14)
    disp.ax_.set_xticklabels(['NO', 'PARTIAL', 'YES'], fontsize=9)
    disp.ax_.set_yticklabels(['NO', 'PARTIAL', 'YES'], fontsize=9)
    print(title)
    print(disp.confusion_matrix)

#plt.show()
plt.savefig('confusion-matrix-scenario1.pdf')


# ### Results: Scenario 1

# In[18]:


def f1score(precision,recall):
    return 2*((precision*recall)/(precision+recall))

print('== SCENARIO 1 - Scores ==')
print('PRECISION:     \n   Gaussian NB  :' + str(np.mean(precision_GNB)) +'  ' + str(np.std(precision_GNB)) +    '\n   Decision Tree:' + str(np.mean(scores_clf_DT_scenario2)) +'  ' + str(np.std(scores_clf_DT_scenario2)) +    '\n   SVM          :' + str(np.mean(scores_clf_SVM_scenario2)) +'  ' + str(np.std(scores_clf_SVM_scenario2)) +    '\n   Random Forest:' + str(np.mean(scores_clf_RF_scenario2))  +'  ' + str(np.std(scores_clf_RF_scenario2)) +    '\n   XGBoost      :' + str(np.mean(scores_clf_XGB_scenario2)) +'  ' + str(np.std(scores_clf_XGB_scenario2)))

print('RECALL:     \n   Gaussian NB  :' + str(np.mean(scores_clf_GNB_scenario2_2)) +'  ' + str(np.std(scores_clf_GNB_scenario2_2)) +    '\n   Decision Tree:' + str(np.mean(scores_clf_DT_scenario2_2)) +'  ' + str(np.std(scores_clf_DT_scenario2_2)) +    '\n   SVM          :' + str(np.mean(scores_clf_SVM_scenario2_2)) +'  ' + str(np.std(scores_clf_SVM_scenario2_2)) +    '\n   Random Forest:' + str(np.mean(scores_clf_RF_scenario2_2))  +'  ' + str(np.std(scores_clf_RF_scenario2_2)) +    '\n   XGBoost      :' + str(np.mean(scores_clf_XGB_scenario2_2)) +'  ' + str(np.std(scores_clf_XGB_scenario2_2)))

print('ACCURACY:     \n   Gaussian NB  :' + str(np.mean(scores_clf_GNB_scenario2_3)) +'  ' + str(np.std(scores_clf_GNB_scenario2_3)) +    '\n   Decision Tree:' + str(np.mean(scores_clf_DT_scenario2_3)) +'  ' + str(np.std(scores_clf_DT_scenario2_3)) +    '\n   SVM          :' + str(np.mean(scores_clf_SVM_scenario2_3)) +'  ' + str(np.std(scores_clf_SVM_scenario2_3)) +    '\n   Random Forest:' + str(np.mean(scores_clf_RF_scenario2_3))  +'  ' + str(np.std(scores_clf_RF_scenario2_3)) +    '\n   XGBoost      :' + str(np.mean(scores_clf_XGB_scenario2_3)) +'  ' + str(np.std(scores_clf_XGB_scenario2_3)))

print('F1:     \n   Gaussian NB  :' + str(np.mean(scores_clf_GNB_scenario2_4)) +'  ' + str(np.std(scores_clf_GNB_scenario2_4)) +    '\n   Decision Tree:' + str(np.mean(scores_clf_DT_scenario2_4)) +'  ' + str(np.std(scores_clf_DT_scenario2_4)) +    '\n   SVM          :' + str(np.mean(scores_clf_SVM_scenario2_4)) +'  ' + str(np.std(scores_clf_SVM_scenario2_4)) +    '\n   Random Forest:' + str(np.mean(scores_clf_RF_scenario2_4))  +'  ' + str(np.std(scores_clf_RF_scenario2_4)) +    '\n   XGBoost      :' + str(np.mean(scores_clf_XGB_scenario2_4)) +'  ' + str(np.std(scores_clf_XGB_scenario2_4)))


# ### Results in Latex syntax: Scenario 1

# ### Function that prints the results in LaTeX table format

# In[19]:


def results_in_latex(scenario):
    print('%%%%  SCENARIO ' + str(scenario) + '  %%%%')
    print('Gaussian NB &')
    F1 = str('{:.4f}'.format(np.mean(scores_clf_GNB_scenario2_4))) + ' $\pm$ ' + str('{:.4f}'.format(np.std(scores_clf_GNB_scenario2_4))) + ' &'
    print(F1)
    PRECISION = str('{:.4f}'.format(np.mean(precision_GNB))) + ' $\pm$ ' + str('{:.4f}'.format(np.std(precision_GNB))) + ' &'
    print(PRECISION)
    RECALL = str('{:.4f}'.format(np.mean(scores_clf_GNB_scenario2_2))) + ' $\pm$ ' + str('{:.4f}'.format(np.std(scores_clf_GNB_scenario2_2)))+ ' &'
    print(RECALL)
    ACCURACY = str('{:.4f}'.format(np.mean(scores_clf_GNB_scenario2_3))) + ' $\pm$ ' + str('{:.4f}'.format(np.std(scores_clf_GNB_scenario2_3)))+ ' \\\\ \hline '
    print(ACCURACY)

    print('Decision Tree & ')
    F1 = str('{:.4f}'.format(np.mean(scores_clf_DT_scenario2_4))) +' $\pm$ ' + str('{:.4f}'.format(np.std(scores_clf_DT_scenario2_4)))+ ' &'
    print(F1)
    PRECISION = str('{:.4f}'.format(np.mean(scores_clf_DT_scenario2))) + ' $\pm$ ' + str('{:.4f}'.format(np.std(scores_clf_DT_scenario2)))+ ' &'
    print(PRECISION)
    RECALL = str('{:.4f}'.format(np.mean(scores_clf_DT_scenario2_2))) + ' $\pm$ ' + str('{:.4f}'.format(np.std(scores_clf_DT_scenario2_2)))+ ' &'
    print(RECALL)
    ACCURACY = str('{:.4f}'.format(np.mean(scores_clf_DT_scenario2_3))) + ' $\pm$ ' + str('{:.4f}'.format(np.std(scores_clf_DT_scenario2_3))) + ' \\\\ \hline '
    print(ACCURACY)

    print('SVM & ')
    F1 = str('{:.4f}'.format(np.mean(scores_clf_SVM_scenario2_4))) +' $\pm$ ' + str('{:.4f}'.format(np.std(scores_clf_SVM_scenario2_4)))+ ' &'
    print(F1)
    PRECISION = str('{:.4f}'.format(np.mean(scores_clf_SVM_scenario2))) + ' $\pm$ ' + str('{:.4f}'.format(np.std(scores_clf_SVM_scenario2)))+ ' &'
    print(PRECISION)
    RECALL = str('{:.4f}'.format(np.mean(scores_clf_SVM_scenario2_2))) + ' $\pm$ ' + str('{:.4f}'.format(np.std(scores_clf_SVM_scenario2_2)))+ ' &'
    print(RECALL)
    ACCURACY = ACCURACY = str('{:.4f}'.format(np.mean(scores_clf_SVM_scenario2_3))) + ' $\pm$ ' + str('{:.4f}'.format(np.std(scores_clf_SVM_scenario2_3))) + ' \\\\ \hline '
    print(ACCURACY)

    print('Random Forest & ')
    F1 = str('{:.4f}'.format(np.mean(scores_clf_RF_scenario2_4)))  +' $\pm$ ' + str('{:.4f}'.format(np.std(scores_clf_RF_scenario2_4))) + ' &'
    print(F1)
    PRECISION = str('{:.4f}'.format(np.mean(scores_clf_RF_scenario2)))  + ' $\pm$ ' + str('{:.4f}'.format(np.std(scores_clf_RF_scenario2)))+ ' &'
    print(PRECISION)
    RECALL = str('{:.4f}'.format(np.mean(scores_clf_RF_scenario2_2)))  + ' $\pm$ ' + str('{:.4f}'.format(np.std(scores_clf_RF_scenario2_2)))+ ' &'
    print(RECALL)
    ACCURACY = str('{:.4f}'.format(np.mean(scores_clf_RF_scenario2_3)))  + ' $\pm$ ' + str('{:.4f}'.format(np.std(scores_clf_RF_scenario2_3))) + ' \\\\ \hline '
    print(ACCURACY)

    print('XGBoost & ')
    F1 = str('{:.4f}'.format(np.mean(scores_clf_XGB_scenario2_4))) +' $\pm$ ' + str('{:.4f}'.format(np.std(scores_clf_XGB_scenario2_4)))+ ' &'
    print(F1)
    PRECISION = str('{:.4f}'.format(np.mean(scores_clf_XGB_scenario2))) + ' $\pm$ ' + str('{:.4f}'.format(np.std(scores_clf_XGB_scenario2)))+ ' &'
    print(PRECISION)
    RECALL = str('{:.4f}'.format(np.mean(scores_clf_XGB_scenario2_2))) + ' $\pm$ ' + str('{:.4f}'.format(np.std(scores_clf_XGB_scenario2_2)))+ ' &'
    print(RECALL)
    ACCURACY = str('{:.4f}'.format(np.mean(scores_clf_XGB_scenario2_3))) + ' $\pm$ ' + str('{:.4f}'.format(np.std(scores_clf_XGB_scenario2_3))) + ' \\\\ \hline '
    print(ACCURACY)


# In[20]:


results_in_latex('1')


# # Scenario 2
# 
# ### Hyperparameter tuning: Scenario 2

# In[21]:


start = time.time()

# Training models
cv = ShuffleSplit(n_splits=cv_nfolds, test_size=0.2, random_state=1)

print('Choosing SVM hyperparameters...')
best_SVM_scenario2 = svc_param_selection(X_tfidf_scenario2, y_coded_scenario2, cv_nfolds)
print('best_SVM_scenario2' + str(best_SVM_scenario2))

print('Choosing XGBoost hyperparameters...')
best_XG_scenario2 = xgboost_param_selection(X_tfidf_scenario2, y_coded_scenario2, cv_nfolds)
print(best_XG_scenario2)

print('Choosing Decision Tree hyperparameters...')
best_DT_scenario2 = dt_param_selection(X_tfidf_scenario2, y_coded_scenario2, cv_nfolds)
print(best_DT_scenario2)

print('Choosing Random Forest hyperparameters...')
best_RF_scenario2 = rf_param_selection(X_tfidf_scenario2, y_coded_scenario2, cv_nfolds)
print(best_RF_scenario2)

# Loading models
clf_GaussianNB_scenario2 = GaussianNB()
clf_DT_scenario2 = best_DT_scenario2
clf_SVM_scenario2 = best_SVM_scenario2
clf_RF_scenario2 = best_RF_scenario2
# clf5=XGBClassifier(learning_rate=0.1, n_estimators=190, max_depth=5, min_child_weight=2, objective="multi:softmax", subsample=0.9, colsample_bytree=0.8, seed=23333)
clf_XGB_scenario2 = best_XG_scenario2

end = time.time()
total_time = end - start
print('Execution time in seconds: ' + str(total_time) )
print('Execution time in minutes: ' + str(total_time/60) )


# ### Model training: Scenario 2

# In[22]:


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, accuracy_score
kf = StratifiedKFold(n_splits=cv_nfolds)

scores_clf_GNB_scenario2=[]
scores_clf_DT_scenario2=[]
scores_clf_SVM_scenario2=[]
scores_clf_RF_scenario2=[]
scores_clf_XGB_scenario2=[]
scores_clf_GNB_scenario2_2=[]
scores_clf_DT_scenario2_2=[]
scores_clf_SVM_scenario2_2=[]
scores_clf_RF_scenario2_2=[]
scores_clf_XGB_scenario2_2=[]
scores_clf_GNB_scenario2_3=[]
scores_clf_DT_scenario2_3=[]
scores_clf_SVM_scenario2_3=[]
scores_clf_RF_scenario2_3=[]
scores_clf_XGB_scenario2_3=[]

# f1
scores_clf_GNB_scenario2_4=[]
scores_clf_DT_scenario2_4=[]
scores_clf_SVM_scenario2_4=[]
scores_clf_RF_scenario2_4=[]
scores_clf_XGB_scenario2_4=[]

start = time.time()

for train_index, test_index in kf.split(X_tfidf_scenario2,y_coded_scenario2):
    X_tfidf_scenario2.toarray(), y_coded_scenario2
    X_train, X_test = X_tfidf_scenario2[train_index], X_tfidf_scenario2[test_index]
    y_train, y_test = y_coded_scenario2[train_index], y_coded_scenario2[test_index]
    
    clf_GaussianNB_scenario2.fit(X_train.toarray(),y_train)
    clf_DT_scenario2.fit(X_train.toarray(),y_train)
    clf_RF_scenario2.fit(X_train.toarray(),y_train)
    clf_SVM_scenario2.fit(X_train,y_train)
    clf_XGB_scenario2.fit(X_train.toarray(),y_train)
    
    y_pred_GNB=clf_GaussianNB_scenario2.predict(X_test.toarray())
    y_pred_DT=clf_DT_scenario2.predict(X_test)
    y_pred_SVM=clf_SVM_scenario2.predict(X_test)
    y_pred_RF=clf_RF_scenario2.predict(X_test)
    y_pred_XGB=clf_XGB_scenario2.predict(X_test)

    scores_clf_GNB_scenario2.append(precision_score(y_test, y_pred_GNB, average='macro'))
    scores_clf_DT_scenario2.append(precision_score(y_test, y_pred_DT, average='macro'))
    scores_clf_SVM_scenario2.append(precision_score(y_test, y_pred_SVM, average='macro'))
    scores_clf_RF_scenario2.append(precision_score(y_test, y_pred_RF, average='macro'))
    scores_clf_XGB_scenario2.append(precision_score(y_test, y_pred_XGB, average='macro'))
    
    scores_clf_GNB_scenario2_2.append(recall_score(y_test, y_pred_GNB, average='macro'))
    scores_clf_DT_scenario2_2.append(recall_score(y_test, y_pred_DT, average='macro'))
    scores_clf_SVM_scenario2_2.append(recall_score(y_test, y_pred_SVM, average='macro'))
    scores_clf_RF_scenario2_2.append(recall_score(y_test, y_pred_RF, average='macro'))
    scores_clf_XGB_scenario2_2.append(recall_score(y_test, y_pred_XGB, average='macro'))
    
    scores_clf_GNB_scenario2_3.append(accuracy_score(y_test, y_pred_GNB))
    scores_clf_DT_scenario2_3.append(accuracy_score(y_test, y_pred_DT))
    scores_clf_SVM_scenario2_3.append(accuracy_score(y_test, y_pred_SVM))
    scores_clf_RF_scenario2_3.append(accuracy_score(y_test, y_pred_RF))
    scores_clf_XGB_scenario2_3.append(accuracy_score(y_test, y_pred_XGB))
    
    scores_clf_GNB_scenario2_4.append(f1_score(y_test, y_pred_GNB, average='macro'))
    scores_clf_DT_scenario2_4.append(f1_score(y_test, y_pred_DT, average='macro'))
    scores_clf_SVM_scenario2_4.append(f1_score(y_test, y_pred_SVM, average='macro'))
    scores_clf_RF_scenario2_4.append(f1_score(y_test, y_pred_RF, average='macro'))
    scores_clf_XGB_scenario2_4.append(f1_score(y_test, y_pred_XGB, average='macro'))
    
end = time.time()
total_time = end - start
print('Execution time in seconds: ' + str(total_time) )
print('Execution time in minutes: ' + str(total_time/60) )


# ### Confusion matrix: Scenario 2

# In[23]:


from sklearn.metrics import plot_confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
class_names = np.unique(y_test)

# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ('Scenario 2: XGBoost model\nNormalized Confusion Matrix', 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(clf_XGB_scenario2, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Greens,
                                 normalize=normalize,
                                 xticks_rotation='horizontal')
    disp.ax_.set_title(title)#, fontsize=14)
    disp.ax_.set_xticklabels(['NO', 'PARTIAL', 'YES'], fontsize=9)
    disp.ax_.set_yticklabels(['NO', 'PARTIAL', 'YES'], fontsize=9)
    print(title)
    print(disp.confusion_matrix)

#plt.show()
plt.savefig('confusion-matrix-scenario2.pdf')


# ### Results: Scenario 2

# In[24]:


print('== SCENARIO 2 - Scores ==')
print('PRECISION:     \n   Gaussian NB  :' + str(np.mean(scores_clf_GNB_scenario2)) +'  ' + str(np.std(scores_clf_GNB_scenario2)) +    '\n   Decision Tree:' + str(np.mean(scores_clf_DT_scenario2)) +'  ' + str(np.std(scores_clf_DT_scenario2)) +    '\n   SVM          :' + str(np.mean(scores_clf_SVM_scenario2)) +'  ' + str(np.std(scores_clf_SVM_scenario2)) +    '\n   Random Forest:' + str(np.mean(scores_clf_RF_scenario2))  +'  ' + str(np.std(scores_clf_RF_scenario2)) +    '\n   XGBoost      :' + str(np.mean(scores_clf_XGB_scenario2)) +'  ' + str(np.std(scores_clf_XGB_scenario2)))

print('RECALL:     \n   Gaussian NB  :' + str(np.mean(scores_clf_GNB_scenario2_2)) +'  ' + str(np.std(scores_clf_GNB_scenario2_2)) +    '\n   Decision Tree:' + str(np.mean(scores_clf_DT_scenario2_2)) +'  ' + str(np.std(scores_clf_DT_scenario2_2)) +    '\n   SVM          :' + str(np.mean(scores_clf_SVM_scenario2_2)) +'  ' + str(np.std(scores_clf_SVM_scenario2_2)) +    '\n   Random Forest:' + str(np.mean(scores_clf_RF_scenario2_2))  +'  ' + str(np.std(scores_clf_RF_scenario2_2)) +    '\n   XGBoost      :' + str(np.mean(scores_clf_XGB_scenario2_2)) +'  ' + str(np.std(scores_clf_XGB_scenario2_2)))

print('ACCURACY:     \n   Gaussian NB  :' + str(np.mean(scores_clf_GNB_scenario2_3)) +'  ' + str(np.std(scores_clf_GNB_scenario2_3)) +    '\n   Decision Tree:' + str(np.mean(scores_clf_DT_scenario2_3)) +'  ' + str(np.std(scores_clf_DT_scenario2_3)) +    '\n   SVM          :' + str(np.mean(scores_clf_SVM_scenario2_3)) +'  ' + str(np.std(scores_clf_SVM_scenario2_3)) +    '\n   Random Forest:' + str(np.mean(scores_clf_RF_scenario2_3))  +'  ' + str(np.std(scores_clf_RF_scenario2_3)) +    '\n   XGBoost      :' + str(np.mean(scores_clf_XGB_scenario2_3)) +'  ' + str(np.std(scores_clf_XGB_scenario2_3)))

print('F1:     \n   Gaussian NB  :' + str(np.mean(scores_clf_GNB_scenario2_4)) +'  ' + str(np.std(scores_clf_GNB_scenario2_4)) +    '\n   Decision Tree:' + str(np.mean(scores_clf_DT_scenario2_4)) +'  ' + str(np.std(scores_clf_DT_scenario2_4)) +    '\n   SVM          :' + str(np.mean(scores_clf_SVM_scenario2_4)) +'  ' + str(np.std(scores_clf_SVM_scenario2_4)) +    '\n   Random Forest:' + str(np.mean(scores_clf_RF_scenario2_4))  +'  ' + str(np.std(scores_clf_RF_scenario2_4)) +    '\n   XGBoost      :' + str(np.mean(scores_clf_XGB_scenario2_4)) +'  ' + str(np.std(scores_clf_XGB_scenario2_4)))


# In[25]:


results_in_latex('2')


# # Scenario 3
# 
# ### Hyperparameter tuning: Scenario 3
# 
# This XGBoost cannot be used for two-variables prediction, so XGBoost was removed from this scenario.

# In[26]:


start = time.time()

# Training models
cv = ShuffleSplit(n_splits=cv_nfolds, test_size=0.2, random_state=1)

print('Choosing SVM hyperparameters...')
best_SVM_scenario3 = svc_param_selection(X_tfidf_scenario3, y_coded_scenario3, cv_nfolds)
print('best_SVM_scenario3' + str(best_SVM_scenario3))

print('Choosing Random Forest hyperparameters...')
best_RF_scenario3 = rf_param_selection(X_tfidf_scenario3, y_coded_scenario3, cv_nfolds)
print(best_RF_scenario3)
# this XGBoost cannot be used for two-variables prediction
# print('Choosing XGBoost hyperparameters...')
# best_XG_scenario3 = xgboost_param_selection(X_tfidf_scenario3, y_coded_scenario3, cv_nfolds)
# print(best_XG_scenario3)

# Loading models
clf_GaussianNB_scenario3 = GaussianNB()
clf_DT_scenario3 = DecisionTreeClassifier()
clf_SVM_scenario3 = best_SVM_scenario3
clf_RF_scenario3 = best_RF_scenario3
# clf5=XGBClassifier(learning_rate=0.1, n_estimators=190, max_depth=5, min_child_weight=2, objective="multi:softmax", subsample=0.9, colsample_bytree=0.8, seed=23333)
# clf_XGB_scenario3 = best_XG_scenario3

end = time.time()
total_time = end - start
print('Execution time in seconds: ' + str(total_time) )
print('Execution time in minutes: ' + str(total_time/60) )


# ### Model training: Scenario 3

# In[27]:


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, accuracy_score
kf = StratifiedKFold(n_splits=cv_nfolds)

#Xgboost can be used for binary classification, we can use the objective funcion `"binary:logistic"`,or ommit the objective parameter for Gridsearch.

def xgboost_param_selection(X, y, nfolds):
    param_grid = {'objective':['binary:logistic'], #
              'learning_rate': [0.3], #so called `eta` value
              'max_depth': [6],
              'min_child_weight': [2,3],
              'silent': [1],
              'subsample': [1],
              'colsample_bytree': [1],
              'n_estimators': [100], #number of trees, change it to 1000 for better results
              'missing':[-999]}
    grid_search = GridSearchCV(XGBClassifier(), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_estimator_
    return grid_search.best_estimator_ 

num_folds=5
print('Chossing XGBoost hyperparameters...')
best_XG_scenario3 = xgboost_param_selection(X_tfidf_scenario3, y_coded_scenario3, num_folds)
print(best_XG_scenario3)

scores_clf_GNB_scenario2=[]
scores_clf_DT_scenario2=[]
scores_clf_SVM_scenario2=[]
scores_clf_RF_scenario2=[]
scores_clf_XGB_scenario2=[]
scores_clf_GNB_scenario2_2=[]
scores_clf_DT_scenario2_2=[]
scores_clf_SVM_scenario2_2=[]
scores_clf_RF_scenario2_2=[]
scores_clf_XGB_scenario2_2=[]
scores_clf_GNB_scenario2_3=[]
scores_clf_DT_scenario2_3=[]
scores_clf_SVM_scenario2_3=[]
scores_clf_RF_scenario2_3=[]
scores_clf_XGB_scenario2_3=[]

# f1
scores_clf_GNB_scenario2_4=[]
scores_clf_DT_scenario2_4=[]
scores_clf_SVM_scenario2_4=[]
scores_clf_RF_scenario2_4=[]
scores_clf_XGB_scenario2_4=[]

start = time.time()

for train_index, test_index in kf.split(X_tfidf_scenario3,y_coded_scenario3):
    X_tfidf_scenario3.toarray(), y_coded_scenario3
    X_train, X_test = X_tfidf_scenario3[train_index], X_tfidf_scenario3[test_index]
    y_train, y_test = y_coded_scenario3[train_index], y_coded_scenario3[test_index]
    
    clf_GaussianNB_scenario3.fit(X_train.toarray(),y_train)
    clf_DT_scenario3.fit(X_train.toarray(),y_train)
    clf_RF_scenario3.fit(X_train.toarray(),y_train)
    clf_SVM_scenario3.fit(X_train,y_train)
    clf_XGB_scenario3.fit(X_train.toarray(),y_train)
    
    y_pred_GNB=clf_GaussianNB_scenario3.predict(X_test.toarray())
    y_pred_DT=clf_DT_scenario3.predict(X_test)
    y_pred_SVM=clf_SVM_scenario3.predict(X_test)
    y_pred_RF=clf_RF_scenario3.predict(X_test)
    y_pred_XGB=clf_XGB_scenario2.predict(X_test)

    scores_clf_GNB_scenario2.append(precision_score(y_test, y_pred_GNB, average='macro'))
    scores_clf_DT_scenario2.append(precision_score(y_test, y_pred_DT, average='macro'))
    scores_clf_SVM_scenario2.append(precision_score(y_test, y_pred_SVM, average='macro'))
    scores_clf_RF_scenario2.append(precision_score(y_test, y_pred_RF, average='macro'))
    scores_clf_XGB_scenario2.append(precision_score(y_test, y_pred_XGB, average='macro'))
    
    scores_clf_GNB_scenario2_2.append(recall_score(y_test, y_pred_GNB, average='macro'))
    scores_clf_DT_scenario2_2.append(recall_score(y_test, y_pred_DT, average='macro'))
    scores_clf_SVM_scenario2_2.append(recall_score(y_test, y_pred_SVM, average='macro'))
    scores_clf_RF_scenario2_2.append(recall_score(y_test, y_pred_RF, average='macro'))
    scores_clf_XGB_scenario2_2.append(recall_score(y_test, y_pred_XGB, average='macro'))
    
    scores_clf_GNB_scenario2_3.append(accuracy_score(y_test, y_pred_GNB))
    scores_clf_DT_scenario2_3.append(accuracy_score(y_test, y_pred_DT))
    scores_clf_SVM_scenario2_3.append(accuracy_score(y_test, y_pred_SVM))
    scores_clf_RF_scenario2_3.append(accuracy_score(y_test, y_pred_RF))
    scores_clf_XGB_scenario2_3.append(accuracy_score(y_test, y_pred_XGB))
    
    scores_clf_GNB_scenario2_4.append(f1_score(y_test, y_pred_GNB, average='macro'))
    scores_clf_DT_scenario2_4.append(f1_score(y_test, y_pred_DT, average='macro'))
    scores_clf_SVM_scenario2_4.append(f1_score(y_test, y_pred_SVM, average='macro'))
    scores_clf_RF_scenario2_4.append(f1_score(y_test, y_pred_RF, average='macro'))
    scores_clf_XGB_scenario2_4.append(f1_score(y_test, y_pred_XGB, average='macro'))
    
end = time.time()
total_time = end - start
print('Execution time in seconds: ' + str(total_time) )
print('Execution time in minutes: ' + str(total_time/60) )    


# ### Confusion Matrix: Scenario 3

# In[28]:


from sklearn.metrics import plot_confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
class_names = np.unique(y_test)

# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ('Scenario 3: SVM model\nNormalized Confusion Matrix', 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(clf_SVM_scenario3, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Reds,
                                 normalize=normalize,
                                 xticks_rotation='horizontal')
    disp.ax_.set_title(title)#, fontsize=14)
#     disp.ax_.set_yscale(.2)
    disp.ax_.set_xticklabels(['NO', 'YES'], fontsize=9)
    disp.ax_.set_yticklabels(['NO', 'YES'], fontsize=9)
    print(title)
    print(disp.confusion_matrix)

#plt.show()
plt.savefig('confusion-matrix-scenario3.pdf')


# ### Results: Scenario 3

# In[29]:


print('== SCENARIO 3 - Scores ==')
print('PRECISION:     \n   Gaussian NB  :' + str(np.mean(scores_clf_GNB_scenario2)) +'  ' + str(np.std(scores_clf_GNB_scenario2)) +    '\n   Decision Tree:' + str(np.mean(scores_clf_DT_scenario2)) +'  ' + str(np.std(scores_clf_DT_scenario2)) +    '\n   SVM          :' + str(np.mean(scores_clf_SVM_scenario2)) +'  ' + str(np.std(scores_clf_SVM_scenario2)) +    '\n   Random Forest:' + str(np.mean(scores_clf_RF_scenario2))  +'  ' + str(np.std(scores_clf_RF_scenario2)) )#+\
#     '\n   XGBoost      :' + str(np.mean(scores_clf_XGB_scenario2)) +'  ' + str(np.std(scores_clf_XGB_scenario2)))

print('RECALL:     \n   Gaussian NB  :' + str(np.mean(scores_clf_GNB_scenario2_2)) +'  ' + str(np.std(scores_clf_GNB_scenario2_2)) +    '\n   Decision Tree:' + str(np.mean(scores_clf_DT_scenario2_2)) +'  ' + str(np.std(scores_clf_DT_scenario2_2)) +    '\n   SVM          :' + str(np.mean(scores_clf_SVM_scenario2_2)) +'  ' + str(np.std(scores_clf_SVM_scenario2_2)) +    '\n   Random Forest:' + str(np.mean(scores_clf_RF_scenario2_2))  +'  ' + str(np.std(scores_clf_RF_scenario2_2)) )#+\
#     '\n   XGBoost      :' + str(np.mean(scores_clf_XGB_scenario2_2)) +'  ' + str(np.std(scores_clf_XGB_scenario2_2)))

print('ACCURACY:     \n   Gaussian NB  :' + str(np.mean(scores_clf_GNB_scenario2_3)) +'  ' + str(np.std(scores_clf_GNB_scenario2_3)) +    '\n   Decision Tree:' + str(np.mean(scores_clf_DT_scenario2_3)) +'  ' + str(np.std(scores_clf_DT_scenario2_3)) +    '\n   SVM          :' + str(np.mean(scores_clf_SVM_scenario2_3)) +'  ' + str(np.std(scores_clf_SVM_scenario2_3)) +    '\n   Random Forest:' + str(np.mean(scores_clf_RF_scenario2_3))  +'  ' + str(np.std(scores_clf_RF_scenario2_3)) ) # +\
#     '\n   XGBoost      :' + str(np.mean(scores_clf_XGB_scenario2_3)) +'  ' + str(np.std(scores_clf_XGB_scenario2_3)))

print('F1:     \n   Gaussian NB  :' + str(np.mean(scores_clf_GNB_scenario2_4)) +'  ' + str(np.std(scores_clf_GNB_scenario2_4)) +    '\n   Decision Tree:' + str(np.mean(scores_clf_DT_scenario2_4)) +'  ' + str(np.std(scores_clf_DT_scenario2_4)) +    '\n   SVM          :' + str(np.mean(scores_clf_SVM_scenario2_4)) +'  ' + str(np.std(scores_clf_SVM_scenario2_4)) +    '\n   Random Forest:' + str(np.mean(scores_clf_RF_scenario2_4))  +'  ' + str(np.std(scores_clf_RF_scenario2_4)) ) # +\
#     '\n   XGBoost      :' + str(np.mean(scores_clf_XGB_scenario2_4)) +'  ' + str(np.std(scores_clf_XGB_scenario2_4)))


# In[30]:


results_in_latex('3')


# ---
# # Unanimity Predictions: Scenarios 4 and 5
# 
# ### Data loading and cleaning

# In[31]:


# Loading the labeled decisions
data_unanimity = pd.read_csv(FILE_DECISION_CSV, sep='<=>',header=0)
print('data_unanimity.shape=' + str(data_unanimity.shape) + ' full data set')

# Removing NA values
data_unanimity = data_unanimity.dropna(subset=[data_unanimity.columns[9]])# decision_description
data_unanimity = data_unanimity.dropna(subset=[data_unanimity.columns[11]])# decision_label
print('data_unanimity.shape=' + str(data_unanimity.shape) + ' dropna')
# data_unanimity = data_unanimity.dropna(subset=[data_unanimity.columns[9]]) # decision_description
# data_unanimity = data_unanimity.dropna(subset=[data_unanimity.columns[13]]) # decision_unanimity

# Removing duplicated samples
data_unanimity = data_unanimity.drop_duplicates(subset=[data_unanimity.columns[1]]) # process_number
print('data_unanimity.shape=' + str(data_unanimity.shape) + ' removed duplicated samples by process_number')
data_unanimity = data_unanimity.drop_duplicates(subset=[data_unanimity.columns[9]]) # decision_description
print('data_unanimity.shape=' + str(data_unanimity.shape) + ' removed duplicated samples by decision_description')

# data_unanimity = data_unanimity.drop_duplicates(subset=[data_unanimity.columns[9]]) # decision_description

# Removing decision_description and decision_labels whose values are -1 and -2
# CORRETO data_unanimity.drop(data_unanimity[data_unanimity['decision_unanimity_text'] == str(-1)].index, inplace = True) 
indexNames = data_unanimity[ (data_unanimity['decision_unanimity_text'] == str(-1)) |                            (data_unanimity['decision_unanimity_text'] == str(-2)) |                             (data_unanimity['decision_unanimity'] == str(-1)) |                             (data_unanimity['decision_unanimity'] == str(-2)) ].index
# print('indexNames='+str(len(indexNames)))
data_unanimity.drop(indexNames, inplace=True)
print('data_unanimity.shape=' + str(data_unanimity.shape) + ' removed -1 and -2 unanimity labels')

# Removing not relevant decision labels and decision not properly labeled
data_unanimity = data_unanimity.query('decision_label != "conflito-competencia"')
print('data_unanimity.shape=' + str(data_unanimity.shape) + ' removed decisions labeled as conflito-competencia')
data = data_unanimity.query('decision_label != "prejudicada"')
print('data_unanimity.shape=' + str(data_unanimity.shape) + ' removed decisions labeled as prejudicada')
data = data_unanimity.query('decision_label != "not-cognized"')
print('data_unanimity.shape=' + str(data_unanimity.shape) + ' removed decisions labeled as not-cognized')


# ### Scenario 4 data set

# In[32]:


print('\n== SCENARIO 4 - Data set ==\n')
# Loading data into vectors and print their frequency
X_scenario4=data_unanimity['decision_description']
y_scenario4=data_unanimity['decision_unanimity']
unique, counts = np.unique(y_scenario4, return_counts=True)
dict_scenario4 = dict(zip(unique, counts))
print('Unanimity frequency: ' + str(dict_scenario4) )


# ### Scenario 5 data set

# In[33]:


# As there are much more `unanimity` decisions rather than `not-unanimity` ones, we randomly remove `unanimity` decisions to have the same number of `unanimity` decisions.
# Making equal the number of unanimity and not-unanimity samples
data_unanimity_u = data_unanimity[ data_unanimity.decision_unanimity == 'unanimity' ]
print('data_unanimity_u.shape=' + str(data_unanimity_u.shape) + ' only unanimity-labeled decisions')
data_unanimity_nu = data_unanimity[ data_unanimity.decision_unanimity == 'not-unanimity' ]
print('data_unanimity_nu.shape=' + str(data_unanimity_nu.shape) + ' only not-unanimity-labeled decisions')
rows = np.random.choice(data_unanimity_u.index.values, dict_scenario4['not-unanimity'])
data_unanimity_u = data_unanimity_u.loc[rows]
data_unanimity = data_unanimity_u.merge(data_unanimity_nu, how='outer')
print('data_unanimity.shape=' + str(data_unanimity.shape) + ' data set balanced: same number of unanmity and not-unanimity decisions')

print('\n== SCENARIO 5 - Data set ==\n')
# data_unanimity
X_scenario5=data_unanimity['decision_description']
y_scenario5=data_unanimity['decision_unanimity']
unique, counts = np.unique(y_scenario5, return_counts=True)
dict_scenario5 = dict(zip(unique, counts))
print('Unanimity frequency: ' + str(dict_scenario5) )


# ## Data pre-processing: cleaning, tf-idf, etc.
# 
# Hector, here is your code because it was simpler to not change it since there are only two scenarios (4 and 5).

# In[34]:


#Stemming leaves only the root of the word. 
stemmer = PorterStemmer()
#Create set of stopwords
stopwords = nltk.corpus.stopwords.words('portuguese')

start = time.time()

# scenario 4
X_preprocessed_scenario4=[]
for i in X_scenario4:
    i=i.lower().replace(".","")
    lineanueva=""
    for pal in i.split():
        if pal not in stopwords:
            lineanueva=lineanueva+stemmer.stem(pal)+" "
    X_preprocessed_scenario4.append(lineanueva)
#print (X_preprocessed)
X_preprocessed_scenario4=np.asarray(X_preprocessed_scenario4)

# scenario 5
X_preprocessed_scenario5=[]
for i in X_scenario5:
    i=i.lower().replace(".","")
    lineanueva=""
    for pal in i.split():
        if pal not in stopwords:
            lineanueva=lineanueva+stemmer.stem(pal)+" "
    X_preprocessed_scenario5.append(lineanueva)
#print (X_preprocessed)
X_preprocessed_scenario5=np.asarray(X_preprocessed_scenario5)

end = time.time()
total_time = end - start
print('Execution time in seconds: ' + str(total_time) )
print('Execution time in minutes: ' + str(total_time/60) )

# print(X)


# ## tf-idf
# ### Scenario 4

# In[35]:


start = time.time()

vectorizer_scenario4 = TfidfVectorizer()
vectorizer_scenario4.fit(X_preprocessed_scenario4)
X_tfidf_scenario4 = vectorizer_scenario4.transform(X_preprocessed_scenario4)
# scenario 5
vectorizer_scenario5 = TfidfVectorizer()
vectorizer_scenario5.fit(X_preprocessed_scenario5)
X_tfidf_scenario5 = vectorizer_scenario5.transform(X_preprocessed_scenario5)

# encoding y
le = preprocessing.LabelEncoder()
y_coded_scenario4 = le.fit_transform(y_scenario4)
y_coded_scenario5 = le.fit_transform(y_scenario5)

end = time.time()
total_time = end - start
print('Execution time in seconds: ' + str(total_time) )
print('Execution time in minutes: ' + str(total_time/60) )


# ### hyperparameter tuning

# In[36]:


# def svc_param_selection(X, y, nfolds):
#     Cs = [0.001, 0.01, 0.1, 1, 10]
#     gammas = [0.001, 0.01, 0.1, 1]
#     degree=[1,2,3,4,5,6,7,8,9]
#     kernel=[ 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
#     param_grid = {'C': Cs, 'gamma' : gammas}
#     grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=nfolds, n_jobs=nof_threads)
#     grid_search.fit(X, y)
#     grid_search.best_estimator_
#     return grid_search.best_estimator_ 
# def xgboost_param_selection(X, y, nfolds):
#     param_grid = {'objective':['multi:softmax'],
#               'learning_rate': [0.3], #so called `eta` value
#               'max_depth': [6],
#               'min_child_weight': [1,2,3],
#               'silent': [1],
#               'subsample': [1],
#               'colsample_bytree': [1],
#               'n_estimators': [1000], #number of trees, change it to 1000 for better results
#               'missing':[-999]}
#     grid_search = GridSearchCV(XGBClassifier(), param_grid, cv=nfolds, n_jobs=nof_threads)
#     grid_search.fit(X, y)
#     grid_search.best_estimator_
#     return grid_search.best_estimator_ 

# def dt_param_selection(X, y, nfolds):
#     param_grid = {'min_samples_split' : range(10,500,20),'max_depth': range(1,20,2)}
#     grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=nfolds, n_jobs=nof_threads)
#     grid_search.fit(X, y)
#     grid_search.best_estimator_
#     return grid_search.best_estimator_

# def rf_param_selection(X, y, nfolds):
#     param_grid = {'max_depth': range(1,7), 'n_estimators': range(100,1100,100)}
#     grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=nfolds, n_jobs=nof_threads)
#     grid_search.fit(X, y)
#     grid_search.best_estimator_
#     return grid_search.best_estimator_


# ### Hyperparameter tuning: Scenario 4

# In[37]:


cv = ShuffleSplit(n_splits=cv_nfolds, test_size=0.2, random_state=1)


# In[69]:


start = time.time()

print('Choosing SVM hyperparameters...')
best_SVM_scenario4 = svc_param_selection(X_tfidf_scenario4, y_coded_scenario4, cv_nfolds)
print('best_SVM_scenario4' + str(best_SVM_scenario4))

print('Choosing Decision Tree hyperparameters...')
best_DT_scenario4 = dt_param_selection(X_tfidf_scenario4, y_coded_scenario4, cv_nfolds)
print(best_DT_scenario4)

print('Choosing Random Forest hyperparameters...')
best_RF_scenario4 = rf_param_selection(X_tfidf_scenario4, y_coded_scenario4, cv_nfolds)
print(best_RF_scenario4)

# models scenario 4
clf_GaussianNB_scenario4 = GaussianNB()
clf_DT_scenario4 = best_DT_scenario4
clf_SVM_scenario4 = best_SVM_scenario4
clf_RF_scenario4 = best_RF_scenario4


end = time.time()
total_time = end - start
print('Execution time in seconds: ' + str(total_time) )
print('Execution time in minutes: ' + str(total_time/60) )


# ### Hyperparameter tuning: Scenario 5

# In[107]:


start = time.time()

print('Choosing SVM hyperparameters...')
best_SVM_scenario5 = svc_param_selection(X_tfidf_scenario5, y_coded_scenario5, cv_nfolds)
print('best_SVM_scenario5' + str(best_SVM_scenario5))

print('Choosing Decision Tree hyperparameters...')
best_DT_scenario5 = dt_param_selection(X_tfidf_scenario5, y_coded_scenario5, cv_nfolds)
print(best_DT_scenario5)

print('Choosing Random Forest hyperparameters...')
best_RF_scenario5 = rf_param_selection(X_tfidf_scenario5, y_coded_scenario5, cv_nfolds)
print(best_RF_scenario5)

clf_GaussianNB_scenario5 = GaussianNB()
clf_DT_scenario5 = best_DT_scenario5
clf_SVM_scenario5 = best_SVM_scenario5
clf_RF_scenario5 = best_RF_scenario5

end = time.time()
total_time = end - start
print('Execution time in seconds: ' + str(total_time) )
print('Execution time in minutes: ' + str(total_time/60) )


# ### Model training: Scenario 4

# In[137]:


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, accuracy_score
kf = StratifiedKFold(n_splits=cv_nfolds)

scores_clf_GNB_scenario2=[]
scores_clf_DT_scenario2=[]
scores_clf_SVM_scenario2=[]
scores_clf_RF_scenario2=[]
scores_clf_XGB_scenario2=[]
scores_clf_GNB_scenario2_2=[]
scores_clf_DT_scenario2_2=[]
scores_clf_SVM_scenario2_2=[]
scores_clf_RF_scenario2_2=[]
scores_clf_XGB_scenario2_2=[]
scores_clf_GNB_scenario2_3=[]
scores_clf_DT_scenario2_3=[]
scores_clf_SVM_scenario2_3=[]
scores_clf_RF_scenario2_3=[]
scores_clf_XGB_scenario2_3=[]

# f1
scores_clf_GNB_scenario2_4=[]
scores_clf_DT_scenario2_4=[]
scores_clf_SVM_scenario2_4=[]
scores_clf_RF_scenario2_4=[]
scores_clf_XGB_scenario2_4=[]

start = time.time()

for train_index, test_index in kf.split(X_tfidf_scenario4,y_coded_scenario4):
    X_tfidf_scenario4.toarray(), y_coded_scenario4
    X_train, X_test = X_tfidf_scenario4[train_index], X_tfidf_scenario4[test_index]
    y_train, y_test = y_coded_scenario4[train_index], y_coded_scenario4[test_index]
    
    clf_GaussianNB_scenario4.fit(X_train.toarray(),y_train)
    clf_DT_scenario4.fit(X_train.toarray(),y_train)
    clf_RF_scenario4.fit(X_train.toarray(),y_train)
    clf_SVM_scenario4.fit(X_train,y_train)
    clf_XGB_scenario1.fit(X_train.toarray(),y_train)
    
    y_pred_GNB=clf_GaussianNB_scenario4.predict(X_test.toarray())
    y_pred_DT=clf_DT_scenario4.predict(X_test)
    y_pred_SVM=clf_SVM_scenario4.predict(X_test)
    y_pred_RF=clf_RF_scenario4.predict(X_test)
    y_pred_XGB=clf_XGB_scenario1.predict(X_test)

    scores_clf_GNB_scenario2.append(precision_score(y_test, y_pred_GNB, average='macro'))
    scores_clf_DT_scenario2.append(precision_score(y_test, y_pred_DT, average='macro'))
    scores_clf_SVM_scenario2.append(precision_score(y_test, y_pred_SVM, average='macro'))
    scores_clf_RF_scenario2.append(precision_score(y_test, y_pred_RF, average='macro'))
    scores_clf_XGB_scenario2.append(precision_score(y_test, y_pred_XGB, average='macro'))
    
    scores_clf_GNB_scenario2_2.append(recall_score(y_test, y_pred_GNB, average='macro'))
    scores_clf_DT_scenario2_2.append(recall_score(y_test, y_pred_DT, average='macro'))
    scores_clf_SVM_scenario2_2.append(recall_score(y_test, y_pred_SVM, average='macro'))
    scores_clf_RF_scenario2_2.append(recall_score(y_test, y_pred_RF, average='macro'))
    scores_clf_XGB_scenario2_2.append(recall_score(y_test, y_pred_XGB, average='macro'))
    
    scores_clf_GNB_scenario2_3.append(accuracy_score(y_test, y_pred_GNB))
    scores_clf_DT_scenario2_3.append(accuracy_score(y_test, y_pred_DT))
    scores_clf_SVM_scenario2_3.append(accuracy_score(y_test, y_pred_SVM))
    scores_clf_RF_scenario2_3.append(accuracy_score(y_test, y_pred_RF))
    
    scores_clf_GNB_scenario2_4.append(f1_score(y_test, y_pred_GNB, average='macro'))
    scores_clf_DT_scenario2_4.append(f1_score(y_test, y_pred_DT, average='macro'))
    scores_clf_SVM_scenario2_4.append(f1_score(y_test, y_pred_SVM, average='macro'))
    scores_clf_RF_scenario2_4.append(f1_score(y_test, y_pred_RF, average='macro'))
    scores_clf_XGB_scenario2_4.append(f1_score(y_test, y_pred_XGB, average='macro'))
    
    #scores_clf_XGB_scenario2_3.append(accuracy_score(y_test, y_pred_XGB))
    #scores_clf_GNB_scenario1_2.append(precision_score(y_test, y_pred_GNB, average='macro'))
    #scores_clf_DT_scenario1_2.append(precision_score(y_test, y_pred_DT, average='macro'))
    #scores_clf_SVM_scenario1_2.append(precision_score(y_test, y_pred_SVM, average='macro'))
    #scores_clf_RF_scenario1_2.append(precision_score(y_test, y_pred_RF, average='macro'))
    #scores_clf_XGB_scenario1_2.append(precision_score(y_test, y_pred_XGB, average='macro'))

end = time.time()
total_time = end - start
print('Execution time in seconds: ' + str(total_time) )
print('Execution time in minutes: ' + str(total_time/60) )


# ### Results: Scenario 4

# In[71]:


print('== SCENARIO 4 - Scores ==')
print('PRECISION:     \n   Gaussian NB  :' + str(np.mean(scores_clf_GNB_scenario2)) +'  ' + str(np.std(scores_clf_GNB_scenario2)) +    '\n   Decision Tree:' + str(np.mean(scores_clf_DT_scenario2)) +'  ' + str(np.std(scores_clf_DT_scenario2)) +    '\n   SVM          :' + str(np.mean(scores_clf_SVM_scenario2)) +'  ' + str(np.std(scores_clf_SVM_scenario2)) +    '\n   Random Forest:' + str(np.mean(scores_clf_RF_scenario2))  +'  ' + str(np.std(scores_clf_RF_scenario2)) +    '\n   XGBoost      :' + str(np.mean(scores_clf_XGB_scenario2)) +'  ' + str(np.std(scores_clf_XGB_scenario2)))

print('RECALL:     \n   Gaussian NB  :' + str(np.mean(scores_clf_GNB_scenario2_2)) +'  ' + str(np.std(scores_clf_GNB_scenario2_2)) +    '\n   Decision Tree:' + str(np.mean(scores_clf_DT_scenario2_2)) +'  ' + str(np.std(scores_clf_DT_scenario2_2)) +    '\n   SVM          :' + str(np.mean(scores_clf_SVM_scenario2_2)) +'  ' + str(np.std(scores_clf_SVM_scenario2_2)) +    '\n   Random Forest:' + str(np.mean(scores_clf_RF_scenario2_2))  +'  ' + str(np.std(scores_clf_RF_scenario2_2)) +    '\n   XGBoost      :' + str(np.mean(scores_clf_XGB_scenario2_2)) +'  ' + str(np.std(scores_clf_XGB_scenario2_2)))

print('ACCURACY:     \n   Gaussian NB  :' + str(np.mean(scores_clf_GNB_scenario2_3)) +'  ' + str(np.std(scores_clf_GNB_scenario2_3)) +    '\n   Decision Tree:' + str(np.mean(scores_clf_DT_scenario2_3)) +'  ' + str(np.std(scores_clf_DT_scenario2_3)) +    '\n   SVM          :' + str(np.mean(scores_clf_SVM_scenario2_3)) +'  ' + str(np.std(scores_clf_SVM_scenario2_3)) +    '\n   Random Forest:' + str(np.mean(scores_clf_RF_scenario2_3))  +'  ' + str(np.std(scores_clf_RF_scenario2_3)) +    '\n   XGBoost      :' + str(np.mean(scores_clf_XGB_scenario2_3)) +'  ' + str(np.std(scores_clf_XGB_scenario2_3)))

print('F1:     \n   Gaussian NB  :' + str(np.mean(scores_clf_GNB_scenario2_4)) +'  ' + str(np.std(scores_clf_GNB_scenario2_4)) +    '\n   Decision Tree:' + str(np.mean(scores_clf_DT_scenario2_4)) +'  ' + str(np.std(scores_clf_DT_scenario2_4)) +    '\n   SVM          :' + str(np.mean(scores_clf_SVM_scenario2_4)) +'  ' + str(np.std(scores_clf_SVM_scenario2_4)) +    '\n   Random Forest:' + str(np.mean(scores_clf_RF_scenario2_4))  +'  ' + str(np.std(scores_clf_RF_scenario2_4)) +    '\n   XGBoost      :' + str(np.mean(scores_clf_XGB_scenario2_4)) +'  ' + str(np.std(scores_clf_XGB_scenario2_4)))


# In[72]:


results_in_latex('4')


# In[138]:


# https://matplotlib.org/1.2.1/mpl_examples/pylab_examples/show_colormaps.pdf

from sklearn.metrics import plot_confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
class_names = np.unique(y_test)

# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ('Scenario 4: Decision Tree model\nNormalized Confusion Matrix', 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(clf_DT_scenario4, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Purples,
                                 normalize=normalize,
                                 xticks_rotation='horizontal')
#     plt.text(j, i, "{:0.2f}".format(cm[i, j])
    disp.ax_.set_title(title)#, fontsize=14)
#     disp.ax_.set_yscale(.2)
    disp.ax_.set_xticklabels(['NOT-UNANIMOUS', 'UNANIMOUS'], fontsize=9)
    disp.ax_.set_yticklabels(['NOT-UNANIMOUS', 'UNANIMOUS'], fontsize=9)
    print(title)
    print(disp.confusion_matrix)

#plt.show()
plt.savefig('confusion-matrix-scenario4.pdf')


# In[139]:


# SCENARIO 4
# Workaround to round the data to 2-digits in the previous confusion matrix

import itertools
import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, title='Scenario 4: Decision Tree model\nNormalized Confusion Matrix', cmap=plt.cm.Purples):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(cm.shape[1])
    plt.xticks(tick_marks)
    ax = plt.gca()
    ax.set_xticklabels((ax.get_xticks() +1).astype(str))
    ax.set_xticklabels(['NOT-UNANIMOUS', 'UNANIMOUS'], fontsize=9)
    ax.set_yticklabels(['NOT-UNANIMOUS', 'UNANIMOUS'], fontsize=9)
    
    plt.yticks(tick_marks)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    

cm = np.array([(0.67,0.33), (0,1)])
np.set_printoptions(precision=1) 
print('Confusion matrix, without normalization')
print(cm)
fig, ax = plt.subplots()

plot_confusion_matrix(cm)

# plt.show()
plt.savefig('confusion-matrix-scenario4.pdf')


# ### Model training: Scenario 5

# In[108]:


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, accuracy_score
kf = StratifiedKFold(n_splits=cv_nfolds)

scores_clf_GNB_scenario2=[]
scores_clf_DT_scenario2=[]
scores_clf_SVM_scenario2=[]
scores_clf_RF_scenario2=[]
scores_clf_XGB_scenario2=[]
scores_clf_GNB_scenario2_2=[]
scores_clf_DT_scenario2_2=[]
scores_clf_SVM_scenario2_2=[]
scores_clf_RF_scenario2_2=[]
scores_clf_XGB_scenario2_2=[]
scores_clf_GNB_scenario2_3=[]
scores_clf_DT_scenario2_3=[]
scores_clf_SVM_scenario2_3=[]
scores_clf_RF_scenario2_3=[]
scores_clf_XGB_scenario2_3=[]

# f1
scores_clf_GNB_scenario2_4=[]
scores_clf_DT_scenario2_4=[]
scores_clf_SVM_scenario2_4=[]
scores_clf_RF_scenario2_4=[]
scores_clf_XGB_scenario2_4=[]

start = time.time()

for train_index, test_index in kf.split(X_tfidf_scenario5,y_coded_scenario5):
    X_tfidf_scenario5.toarray(), y_coded_scenario5
    X_train, X_test = X_tfidf_scenario5[train_index], X_tfidf_scenario5[test_index]
    y_train, y_test = y_coded_scenario5[train_index], y_coded_scenario5[test_index]
    
    clf_GaussianNB_scenario5.fit(X_train.toarray(),y_train)
    clf_DT_scenario5.fit(X_train.toarray(),y_train)
    clf_RF_scenario5.fit(X_train.toarray(),y_train)
    clf_SVM_scenario5.fit(X_train,y_train)
    clf_XGB_scenario1.fit(X_train.toarray(),y_train)
    
    y_pred_GNB=clf_GaussianNB_scenario5.predict(X_test.toarray())
    y_pred_DT=clf_DT_scenario5.predict(X_test)
    y_pred_SVM=clf_SVM_scenario5.predict(X_test)
    y_pred_RF=clf_RF_scenario5.predict(X_test)
    y_pred_XGB=clf_XGB_scenario1.predict(X_test)

    scores_clf_GNB_scenario2.append(precision_score(y_test, y_pred_GNB, average='macro'))
    scores_clf_DT_scenario2.append(precision_score(y_test, y_pred_DT, average='macro'))
    scores_clf_SVM_scenario2.append(precision_score(y_test, y_pred_SVM, average='macro'))
    scores_clf_RF_scenario2.append(precision_score(y_test, y_pred_RF, average='macro'))
    scores_clf_XGB_scenario2.append(precision_score(y_test, y_pred_XGB, average='macro'))
    
    scores_clf_GNB_scenario2_2.append(recall_score(y_test, y_pred_GNB, average='macro'))
    scores_clf_DT_scenario2_2.append(recall_score(y_test, y_pred_DT, average='macro'))
    scores_clf_SVM_scenario2_2.append(recall_score(y_test, y_pred_SVM, average='macro'))
    scores_clf_RF_scenario2_2.append(recall_score(y_test, y_pred_RF, average='macro'))
    scores_clf_XGB_scenario2_2.append(recall_score(y_test, y_pred_XGB, average='macro'))
    
    scores_clf_GNB_scenario2_3.append(accuracy_score(y_test, y_pred_GNB))
    scores_clf_DT_scenario2_3.append(accuracy_score(y_test, y_pred_DT))
    scores_clf_SVM_scenario2_3.append(accuracy_score(y_test, y_pred_SVM))
    scores_clf_RF_scenario2_3.append(accuracy_score(y_test, y_pred_RF))
    
    scores_clf_GNB_scenario2_4.append(f1_score(y_test, y_pred_GNB, average='macro'))
    scores_clf_DT_scenario2_4.append(f1_score(y_test, y_pred_DT, average='macro'))
    scores_clf_SVM_scenario2_4.append(f1_score(y_test, y_pred_SVM, average='macro'))
    scores_clf_RF_scenario2_4.append(f1_score(y_test, y_pred_RF, average='macro'))
    scores_clf_XGB_scenario2_4.append(f1_score(y_test, y_pred_XGB, average='macro'))
    
    #scores_clf_XGB_scenario2_3.append(accuracy_score(y_test, y_pred_XGB))
    #scores_clf_GNB_scenario1_2.append(precision_score(y_test, y_pred_GNB, average='macro'))
    #scores_clf_DT_scenario1_2.append(precision_score(y_test, y_pred_DT, average='macro'))
    #scores_clf_SVM_scenario1_2.append(precision_score(y_test, y_pred_SVM, average='macro'))
    #scores_clf_RF_scenario1_2.append(precision_score(y_test, y_pred_RF, average='macro'))
    #scores_clf_XGB_scenario1_2.append(precision_score(y_test, y_pred_XGB, average='macro'))
    
end = time.time()
total_time = end - start
print('Execution time in seconds: ' + str(total_time) )
print('Execution time in minutes: ' + str(total_time/60) )


# ### Results: Scenario 5

# In[109]:


print('== SCENARIO 5 - Scores ==')
print('PRECISION:     \n   Gaussian NB  :' + str(np.mean(scores_clf_GNB_scenario2)) +'  ' + str(np.std(scores_clf_GNB_scenario2)) +    '\n   Decision Tree:' + str(np.mean(scores_clf_DT_scenario2)) +'  ' + str(np.std(scores_clf_DT_scenario2)) +    '\n   SVM          :' + str(np.mean(scores_clf_SVM_scenario2)) +'  ' + str(np.std(scores_clf_SVM_scenario2)) +    '\n   Random Forest:' + str(np.mean(scores_clf_RF_scenario2))  +'  ' + str(np.std(scores_clf_RF_scenario2)) +    '\n   XGBoost      :' + str(np.mean(scores_clf_XGB_scenario2)) +'  ' + str(np.std(scores_clf_XGB_scenario2)))

print('RECALL:     \n   Gaussian NB  :' + str(np.mean(scores_clf_GNB_scenario2_2)) +'  ' + str(np.std(scores_clf_GNB_scenario2_2)) +    '\n   Decision Tree:' + str(np.mean(scores_clf_DT_scenario2_2)) +'  ' + str(np.std(scores_clf_DT_scenario2_2)) +    '\n   SVM          :' + str(np.mean(scores_clf_SVM_scenario2_2)) +'  ' + str(np.std(scores_clf_SVM_scenario2_2)) +    '\n   Random Forest:' + str(np.mean(scores_clf_RF_scenario2_2))  +'  ' + str(np.std(scores_clf_RF_scenario2_2)) +    '\n   XGBoost      :' + str(np.mean(scores_clf_XGB_scenario2_2)) +'  ' + str(np.std(scores_clf_XGB_scenario2_2)))

print('ACCURACY:     \n   Gaussian NB  :' + str(np.mean(scores_clf_GNB_scenario2_3)) +'  ' + str(np.std(scores_clf_GNB_scenario2_3)) +    '\n   Decision Tree:' + str(np.mean(scores_clf_DT_scenario2_3)) +'  ' + str(np.std(scores_clf_DT_scenario2_3)) +    '\n   SVM          :' + str(np.mean(scores_clf_SVM_scenario2_3)) +'  ' + str(np.std(scores_clf_SVM_scenario2_3)) +    '\n   Random Forest:' + str(np.mean(scores_clf_RF_scenario2_3))  +'  ' + str(np.std(scores_clf_RF_scenario2_3)) +    '\n   XGBoost      :' + str(np.mean(scores_clf_XGB_scenario2_3)) +'  ' + str(np.std(scores_clf_XGB_scenario2_3)))

print('F1:     \n   Gaussian NB  :' + str(np.mean(scores_clf_GNB_scenario2_4)) +'  ' + str(np.std(scores_clf_GNB_scenario2_4)) +    '\n   Decision Tree:' + str(np.mean(scores_clf_DT_scenario2_4)) +'  ' + str(np.std(scores_clf_DT_scenario2_4)) +    '\n   SVM          :' + str(np.mean(scores_clf_SVM_scenario2_4)) +'  ' + str(np.std(scores_clf_SVM_scenario2_4)) +    '\n   Random Forest:' + str(np.mean(scores_clf_RF_scenario2_4))  +'  ' + str(np.std(scores_clf_RF_scenario2_4)) +    '\n   XGBoost      :' + str(np.mean(scores_clf_XGB_scenario2_4)) +'  ' + str(np.std(scores_clf_XGB_scenario2_4)))


# In[110]:


results_in_latex('5')


# In[111]:


# # cmap = [Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, 
# BuPu_r, CMRmap, CMRmap_r, Dark2, Dark2_r, GnBu, GnBu_r, Greens, Greens_r, 
# Greys, Greys_r, OrRd, OrRd_r, Oranges, Oranges_r, PRGn, PRGn_r, Paired, Paired_r, 
# Pastel1, Pastel1_r, Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, 
# PuBu_r, PuOr, PuOr_r, PuRd, PuRd_r, Purples, Purples_r, RdBu, RdBu_r, RdGy, RdGy_r, 
# RdPu, RdPu_r, RdYlBu, RdYlBu_r, RdYlGn, RdYlGn_r, Reds, Reds_r, Set1, Set1_r, 
# Set2, Set2_r, Set3, Set3_r, Spectral, Spectral_r, Wistia, Wistia_r, YlGn, YlGnBu, 
# YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r, YlOrRd, YlOrRd_r, afmhot, afmhot_r, autumn,
# autumn_r, binary, binary_r, bone, bone_r, brg, brg_r, bwr, bwr_r, cividis, 
# cividis_r, cool, cool_r, coolwarm, coolwarm_r, copper, copper_r, cubehelix, 
# cubehelix_r, flag, flag_r, gist_earth, gist_earth_r, gist_gray, gist_gray_r,
# gist_heat, gist_heat_r, gist_ncar, gist_ncar_r, gist_rainbow, gist_rainbow_r, 
# gist_stern, gist_stern_r, gist_yarg, gist_yarg_r, gnuplot, gnuplot2, gnuplot2_r, 
# gnuplot_r, gray, gray_r, hot, hot_r, hsv, hsv_r, icefire, icefire_r, inferno, 
# inferno_r, jet, jet_r, magma, magma_r, mako, mako_r, nipy_spectral, nipy_spectral_r,
# ocean, ocean_r, pink, pink_r, plasma, plasma_r, prism, prism_r, rainbow, rainbow_r, 
# rocket, rocket_r, seismic, seismic_r, spring, spring_r, summer, summer_r, tab10, 
# tab10_r, tab20, tab20_r, tab20b, tab20b_r, tab20c, tab20c_r, terrain, terrain_r, 
# viridis, viridis_r, vlag, vlag_r, winter, winter_r]

from sklearn.metrics import plot_confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
class_names = np.unique(y_test)

# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ('Scenario 5: Random Forest model\nNormalized Confusion Matrix', 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(clf_RF_scenario5, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Greys,
                                 normalize=normalize,
                                 xticks_rotation='horizontal')
    disp.ax_.set_title(title)#, fontsize=14)
#     disp.ax_.set_yscale(.2)
    disp.ax_.set_xticklabels(['NOT-UNANIMOUS', 'UNANIMOUS'], fontsize=9)
    disp.ax_.set_yticklabels(['NOT-UNANIMOUS', 'UNANIMOUS'], fontsize=9)
    print(title)
    print(disp.confusion_matrix)

#plt.show()
plt.savefig('confusion-matrix-scenario5.pdf')


# In[140]:


# SCENARIO 5
# Workaround to round the data to 2-digits in the previous confusion matrix

import itertools
import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, title='Scenario 5: Random Forest model\nNormalized Confusion Matrix', cmap=plt.cm.Greys):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(cm.shape[1])
    plt.xticks(tick_marks)
    ax = plt.gca()
    ax.set_xticklabels((ax.get_xticks() +1).astype(str))
    ax.set_xticklabels(['NOT-UNANIMOUS', 'UNANIMOUS'], fontsize=9)
    ax.set_yticklabels(['NOT-UNANIMOUS', 'UNANIMOUS'], fontsize=9)
    
    plt.yticks(tick_marks)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    

cm = np.array([(0.78,0.22), (0,1)])
np.set_printoptions(precision=1) 
print('Confusion matrix, without normalization')
print(cm)
fig, ax = plt.subplots()

plot_confusion_matrix(cm)

# plt.show()
plt.savefig('confusion-matrix-scenario5.pdf')
#BERT Model
#Scenario 1

# Import the pytorch library
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

#We are going to generate a CustomDataset (a way to load data in batches)
#For pytorch, it will facilitate the training
class CustomDataset(Dataset):
    def __init__(self, sentences, targets , tokenizer, max_len):
        self.tokenizer = tokenizer
        self.sentences = sentences
        self.targets = targets
        self.max_len = max_len
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, index):
        sentence = str(self.sentences[index])
        sentence = " ".join(sentence.split())
        inputs = self.tokenizer.encode_plus(
            sentence,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids'] #Ids of volcabulary
        mask = inputs['attention_mask'] #Masks to define where the attention should see
        token_type_ids = inputs["token_type_ids"] #required but not used
        
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }
    
from torch import cuda
device = 'cuda:0' if cuda.is_available() else 'cpu'

# We define constants for the execution of our code
MAX_LEN = 200 # 2470
TRAIN_BATCH_SIZE = 8
TEST_BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 1e-05

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
kf = StratifiedKFold(n_splits=5) # n-fold = 5

#Let's define a modified Bert Class
class BERTClass(torch.nn.Module):
    def __init__(self,b_model):
        super(BERTClass, self).__init__()
        self.l1 = b_model # We define that the first layer is the bert model that receives by parameter
        self.l2 = torch.nn.Dropout(0.3) # At the output of bert, we are going to apply a dropout of 0.3
        #self.l3 = torch.nn.Linear(768, 3) #Finalmente vamos a definir una capa densa (combinación lineal) de la dimension de bert a nuestra neurona de salida
        self.l3 = torch.nn.Linear(1024, 3) #Finally we are going to define a dense layer (linear combination) of the bert dimension to our output neuron
        self.out_activation = torch.nn.Sigmoid() 
    
    def forward(self, ids, mask, token_type_ids):
        #We do the forward pass through BERT, dropout and the dense layer
        output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids)[1]
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        #output = self.out_activation(output)
        return output

def train(epoch,model,training_loader,optimizer):
    model.train()
    loss_acum=0
    for iters , data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.long)

        outputs = model(ids, mask, token_type_ids)
        outputs = outputs.squeeze(1)
        #print(outputs,targets)
        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        loss_acum += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss_acum/iters
    
    
def validation(epoch,model,testing_loader):
    model.eval()
    loss_acum=0
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for iters, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.long)
            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            loss = loss_fn(outputs.squeeze(1), targets)
            loss_acum += loss.item()
            fin_outputs.extend(torch.softmax(outputs,1).squeeze(1).cpu().detach().numpy()) 
    
    return loss_acum/iters ,np.array(fin_outputs), np.array(fin_targets)
#We define our loss function
def loss_fn(outputs, targets):
    return torch.nn.CrossEntropyLoss()(outputs, targets)

test_metrics = []

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': TEST_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_metrics

#Scenario 2

test_metrics = []

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': TEST_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }


from tqdm.notebook import tqdm
for n_exp in tqdm(range(5)):
    best_losses = []
    best_metrics = []
    n_split=0
    for train_idx,test_idx in kf.split(X_scenario2,y_coded_scenario2):
        n_split+=1
        temp_X, temp_y = X_scenario2[train_idx], y_coded_scenario2[train_idx]
        train_X, val_X, train_y, val_y =  train_test_split(temp_X,temp_y, test_size=0.1, shuffle=True)
        test_X, test_y = X_scenario2[test_idx], y_coded_scenario2[test_idx]
        
        training_set = CustomDataset(train_X.values, train_y, tokenizer, MAX_LEN)
        validation_set = CustomDataset(val_X.values, val_y, tokenizer, MAX_LEN)
        test_set = CustomDataset(test_X.values, test_y, tokenizer, MAX_LEN)
        
        training_loader = DataLoader(training_set, **train_params)
        validation_loader = DataLoader(validation_set, **train_params)
        testing_loader = DataLoader(test_set, **test_params)
        
        bert_model = BertModel.from_pretrained('neuralmind/bert-large-portuguese-cased')
        model_b = BERTClass(bert_model)
        model_b.to(device)
        optimizer = torch.optim.Adam(params =  model_b.parameters(), lr=LEARNING_RATE)
        aux_loss = []
        
        aux_loss = []
        for epoch in tqdm(range(EPOCHS),desc="Train split {}".format(n_split)):
            epoch_loss = train(epoch,model_b,training_loader,optimizer,)
            val_loss, outputs, targets = validation(1,model_b,validation_loader)
            aux_loss.append(val_loss)
            outputs_cat = np.argmax(outputs, 1)
            best_metrics.append({'n_split':n_split,'epoch':epoch,
                'accuracy':metrics.accuracy_score(targets, outputs_cat),
                'f1_micro':metrics.f1_score(targets, outputs_cat, average='micro'),
                'f1_macro':metrics.f1_score(targets, outputs_cat, average='macro'),
                'precision_macro':metrics.precision_score(targets, outputs_cat, average='macro'),
                'recall_macro':metrics.recall_score(targets, outputs_cat, average='macro'),
                                })
        
        best_losses.append((np.argmin(aux_loss),np.min(aux_loss)))
        new_epoch = np.argmin(aux_loss) + 1
        
        
        training_set = CustomDataset(temp_X.values,temp_y, tokenizer, MAX_LEN) #Whole Training Set
        training_loader = DataLoader(training_set, **train_params)
        
        bert_model = BertModel.from_pretrained('neuralmind/bert-large-portuguese-cased')
        model_b = BERTClass(bert_model)
        model_b.to(device)
        optimizer = torch.optim.Adam(params =  model_b.parameters(), lr=LEARNING_RATE)
        
        for epoch in tqdm(range(new_epoch),desc="Test split {}".format(n_split)):
            train(epoch,model_b,training_loader,optimizer,)
        
        test_loss, outputs, targets = validation(1,model_b,testing_loader)
        outputs_cat = np.argmax(outputs, 1)
        test_metrics.append({'n_exp':n_exp,'n_split':n_split,'epoch':new_epoch,
            'accuracy':metrics.accuracy_score(targets, outputs_cat),
            'f1_micro':metrics.f1_score(targets, outputs_cat, average='micro'),
            'f1_macro':metrics.f1_score(targets, outputs_cat, average='macro'),
            'precision_macro':metrics.precision_score(targets, outputs_cat, average='macro'),
            'recall_macro':metrics.recall_score(targets, outputs_cat, average='macro'),
                            })
#Scenario 3

        #Modified Bert Class
class BERTClass(torch.nn.Module):
    def __init__(self,b_model):
        super(BERTClass, self).__init__()
        self.l1 = b_model # We define that the first layer is the bert model that receives by parameter
        self.l2 = torch.nn.Dropout(0.3) # At the output of bert, we are going to apply a dropout of 0.3
        #self.l3 = torch.nn.Linear(768, 3) #Finalmente vamos a definir una capa densa (combinación lineal) de la dimension de bert a nuestra neurona de salida
        self.l3 = torch.nn.Linear(1024, 1) #Finally we are going to define a dense layer (linear combination) of the bert dimension to our output neuron
        self.out_activation = torch.nn.Sigmoid() 
    
    def forward(self, ids, mask, token_type_ids):
        #We do the forward pass through BERT, dropout and the dense layer
        output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids)[1]
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        #output = self.out_activation(output)
        return output
#Loss function
def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

def train(epoch,model,training_loader,optimizer):
    model.train()
    loss_acum=0
    for iters , data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)

        outputs = model(ids, mask, token_type_ids)
        outputs = outputs.squeeze(1)
        #print(outputs,targets)
        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        loss_acum += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss_acum/iters
def validation(epoch,model,testing_loader):
    model.eval()
    loss_acum=0
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for iters, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            loss = loss_fn(outputs.squeeze(1), targets)
            loss_acum += loss.item()
            fin_outputs.extend(torch.sigmoid(outputs).squeeze(1).cpu().detach().numpy()) 
    
    return loss_acum/iters ,np.array(fin_outputs), np.array(fin_targets)
test_metrics = []

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': TEST_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }


from tqdm.notebook import tqdm
for n_exp in tqdm(range(5)):
    best_losses = []
    best_metrics = []
    n_split=0
    for train_idx,test_idx in kf.split(X_scenario3,y_coded_scenario3):
        n_split+=1
        temp_X, temp_y = X_scenario3[train_idx], y_coded_scenario3[train_idx]
        train_X, val_X, train_y, val_y =  train_test_split(temp_X,temp_y, test_size=0.1, shuffle=True)
        test_X, test_y = X_scenario3[test_idx], y_coded_scenario3[test_idx]
        
        training_set = CustomDataset(train_X.values, train_y, tokenizer, MAX_LEN)
        validation_set = CustomDataset(val_X.values, val_y, tokenizer, MAX_LEN)
        test_set = CustomDataset(test_X.values, test_y, tokenizer, MAX_LEN)
        
        training_loader = DataLoader(training_set, **train_params)
        validation_loader = DataLoader(validation_set, **train_params)
        testing_loader = DataLoader(test_set, **test_params)
        
        bert_model = BertModel.from_pretrained('neuralmind/bert-large-portuguese-cased')
        model_b = BERTClass(bert_model)
        model_b.to(device)
        optimizer = torch.optim.Adam(params =  model_b.parameters(), lr=LEARNING_RATE)
        aux_loss = []
        
        aux_loss = []
        for epoch in tqdm(range(EPOCHS),desc="Train split {}".format(n_split)):
            epoch_loss = train(epoch,model_b,training_loader,optimizer,)
            val_loss, outputs, targets = validation(1,model_b,validation_loader)
            aux_loss.append(val_loss)
            outputs_bin = (outputs >= 0.5)
            best_metrics.append({'n_split':n_split,'epoch':epoch,
                'accuracy':metrics.accuracy_score(targets, outputs_bin),
                'f1_micro':metrics.f1_score(targets, outputs_bin, average='micro'),
                'f1_macro':metrics.f1_score(targets, outputs_bin, average='macro'),
                'precision_macro':metrics.precision_score(targets, outputs_bin, average='macro'),
                'recall_macro':metrics.recall_score(targets, outputs_bin, average='macro'),
                                })
        
        best_losses.append((np.argmin(aux_loss),np.min(aux_loss)))
        new_epoch = np.argmin(aux_loss) + 1
        
        
        training_set = CustomDataset(temp_X.values,temp_y, tokenizer, MAX_LEN) #Whole Training Set
        training_loader = DataLoader(training_set, **train_params)
        
        bert_model = BertModel.from_pretrained('neuralmind/bert-large-portuguese-cased')
        model_b = BERTClass(bert_model)
        model_b.to(device)
        optimizer = torch.optim.Adam(params =  model_b.parameters(), lr=LEARNING_RATE)
        
        for epoch in tqdm(range(new_epoch),desc="Test split {}".format(n_split)):
            train(epoch,model_b,training_loader,optimizer,)
        
        test_loss, outputs, targets = validation(1,model_b,testing_loader)
        outputs_bin = (outputs >= 0.5)
        test_metrics.append({'n_exp':n_exp,'n_split':n_split,'epoch':new_epoch,
            'accuracy':metrics.accuracy_score(targets, outputs_bin),
            'f1_micro':metrics.f1_score(targets, outputs_bin, average='micro'),
            'f1_macro':metrics.f1_score(targets, outputs_bin, average='macro'),
            'precision_macro':metrics.precision_score(targets, outputs_bin, average='macro'),
            'recall_macro':metrics.recall_score(targets, outputs_bin, average='macro'),
                            })
test_metrics

#BILSTM
#Scneario 1
from sklearn.preprocessing import OneHotEncoder

le1 = preprocessing.LabelEncoder()
le2 = preprocessing.LabelEncoder()
le3  = preprocessing.LabelEncoder()
y_coded_scenario1=le1.fit_transform(y_scenario1)
y_coded_scenario2=le2.fit_transform(y_scenario2)

oe1 = OneHotEncoder()
oe2 = OneHotEncoder()

y_coded_scenario1 = oe1.fit_transform(y_coded_scenario1.reshape(-1, 1)).toarray()
y_coded_scenario2 = oe2.fit_transform(y_coded_scenario2.reshape(-1, 1)).toarray()


print(list(le2.inverse_transform([0, 1, 2])))
y_coded_scenario3=le3.fit_transform(y_scenario3)

y_coded_scenario3

from gensim.models import KeyedVectors
wv = KeyedVectors.load_word2vec_format('skip_s300.txt')

import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
tf.__version__

from tensorflow import keras
from tensorflow.keras import layers

physical_devices = tf.config.list_physical_devices('GPU')
physical_devices
tf.config.experimental.set_memory_growth(physical_devices[0], True)
# Constants
MAX_LEN = 100 # 2470
TRAIN_BATCH_SIZE = 16
TEST_BATCH_SIZE = 16
EPOCHS = 20

MAX_FEATURES = 8000  # First 8000 words to generate dictionary

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score


def get_model(hidden_dim, output_dim, maxlen, max_features, emb_matrix,last_activation = 'softmax'):
    #Input layer, which will receive the interos arrangements (vocabulary indexes)
    inputs = keras.Input(shape=(maxlen,), dtype="int32")
    # We transform each index into its corresponding vector of words
    # But this time, using the weights parameter, we initialize this vector layer
    # With the weights extracted from the pre-trained model
    # Besides, we define that this layer is not trainable, that is, the values ​​of the weights
    # Do not adjust in each iteration of training
    x = layers.Embedding(max_features + 1, 300,weights=[emb_matrix], trainable=False)(inputs)
    # Add a layer of LSTM
    x = layers.Bidirectional(layers.LSTM(hidden_dim))(x)
   
    # We add the output layer, 1 output neuron because it is binary classification
    # In addition, we use the sigmoid activation function to return the probability
    outputs = layers.Dense(output_dim, activation=last_activation)(x)

#We generate the model
    model = keras.Model(inputs, outputs)
    #model1.summary()
    return model
import gc
import pickle
test_metrics = []


checkpoint_name = 'BILSTM_Scenario1.h5'
checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_name, verbose=0, 
                                             monitor='val_accuracy',save_best_only=True, mode='auto')
kf = StratifiedKFold(n_splits=5) # n-fold = 5
from tqdm.notebook import tqdm
for n_exp in tqdm(range(5)):
    best_losses = []
    best_metrics = []
    n_split=0
    kf = StratifiedKFold(n_splits=5) # n-fold = 5
    for train_idx,test_idx in tqdm(kf.split(X_scenario1,y_coded_scenario1.argmax(1)),total=5):
        n_split+=1
        temp_X, temp_y = X_scenario1[train_idx], y_coded_scenario1[train_idx]
        train_X, val_X, train_y, val_y =  train_test_split(temp_X,temp_y, test_size=0.1, shuffle=True)
        test_X, test_y = X_scenario1[test_idx], y_coded_scenario1[test_idx]
        
        tokenizer = keras.preprocessing.text.Tokenizer(num_words = MAX_FEATURES, oov_token='<unk>', )
        tokenizer.fit_on_texts(train_X)
        
        train_X_seq = tokenizer.texts_to_sequences(train_X)
        val_X_seq = tokenizer.texts_to_sequences(val_X)
        test_X_seq = tokenizer.texts_to_sequences(test_X)
        
        train_X_seq_padded = keras.preprocessing.sequence.pad_sequences(train_X_seq,maxlen= MAX_LEN)
        val_X_seq_padded = keras.preprocessing.sequence.pad_sequences(val_X_seq,maxlen= MAX_LEN)
        test_X_seq_padded = keras.preprocessing.sequence.pad_sequences(test_X_seq,maxlen= MAX_LEN)
        
        emb_matrix = np.random.rand(MAX_FEATURES + 1,300)
        
        for i in range(1, MAX_FEATURES + 1):
#We get the word corresponding to index i
            word = tokenizer.index_word[i]
            if word in wv: #We ask if the word is in the word vector model
                emb_matrix[i] = wv[word] # We assign the value to the row to the word vector
        
        model = get_model(hidden_dim=128, output_dim=3, maxlen=MAX_LEN, 
                          max_features=MAX_FEATURES,emb_matrix=emb_matrix,last_activation='softmax')
        model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
        model.fit(train_X_seq_padded, train_y, batch_size=TRAIN_BATCH_SIZE, 
                  epochs=EPOCHS, callbacks=[checkpoint],verbose=0, 
                  validation_data=(val_X_seq_padded, val_y))
        
        model.load_weights(checkpoint_name)
        y_pred = model.predict(test_X_seq_padded,batch_size=TEST_BATCH_SIZE,verbose=0)
        
        y_pred = y_pred.argmax(1)
        y_pred = y_pred.astype(np.int8)
        
        accuracy = metrics.accuracy_score(test_y.argmax(1), y_pred)
        f1_score_micro = metrics.f1_score(test_y.argmax(1), y_pred, average='micro')
        f1_score_macro = metrics.f1_score(test_y.argmax(1), y_pred, average='macro')
        precision_macro = metrics.precision_score(test_y.argmax(1), y_pred, average='macro')
        recall_macro = metrics.recall_score(test_y.argmax(1), y_pred, average='macro')
        
        test_metrics.append({'n_exp':n_exp,'n_split':n_split,'epoch':15,
            'accuracy':accuracy,
            'f1_micro':f1_score_micro,
            'f1_macro':f1_score_macro,
            'precision_macro':precision_macro,
            'recall_macro':recall_macro,
                            })
        
        del model
        for _ in range(5): gc.collect()
        

        f = open("BILSTM_Results/scenario1_test_metrics.pkl","wb")
        pickle.dump(test_metrics,f)
        f.close()
        #print(test_metrics)
#Scenario 2
test_metrics = []


checkpoint_name = 'BILSTM_Scenario2.h5'
checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_name, verbose=0, 
                                             monitor='val_accuracy',save_best_only=True, mode='auto')
kf = StratifiedKFold(n_splits=5) # n-fold = 5
from tqdm.notebook import tqdm
for n_exp in tqdm(range(5)):
    best_losses = []
    best_metrics = []
    n_split=0
    kf = StratifiedKFold(n_splits=5) # n-fold = 5
    for train_idx,test_idx in tqdm(kf.split(X_scenario2,y_coded_scenario2.argmax(1)),total=5):
        n_split+=1
        temp_X, temp_y = X_scenario2[train_idx], y_coded_scenario2[train_idx]
        train_X, val_X, train_y, val_y =  train_test_split(temp_X,temp_y, test_size=0.1, shuffle=True)
        test_X, test_y = X_scenario2[test_idx], y_coded_scenario2[test_idx]
        
        tokenizer = keras.preprocessing.text.Tokenizer(num_words = MAX_FEATURES, oov_token='<unk>', )
        tokenizer.fit_on_texts(train_X)
        
        train_X_seq = tokenizer.texts_to_sequences(train_X)
        val_X_seq = tokenizer.texts_to_sequences(val_X)
        test_X_seq = tokenizer.texts_to_sequences(test_X)
        
        train_X_seq_padded = keras.preprocessing.sequence.pad_sequences(train_X_seq,maxlen= MAX_LEN)
        val_X_seq_padded = keras.preprocessing.sequence.pad_sequences(val_X_seq,maxlen= MAX_LEN)
        test_X_seq_padded = keras.preprocessing.sequence.pad_sequences(test_X_seq,maxlen= MAX_LEN)
        
        emb_matrix = np.random.rand(MAX_FEATURES + 1,300)
        
        for i in range(1, MAX_FEATURES + 1):
#We get the word corresponding to index i
            word = tokenizer.index_word[i]
            if word in wv: #We ask if the word is in the word vector model
                emb_matrix[i] = wv[word] # We assign the value to the row to the word vector
        
        model = get_model(hidden_dim=128, output_dim=3, maxlen=MAX_LEN, 
                          max_features=MAX_FEATURES,emb_matrix=emb_matrix,last_activation='softmax')
        model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
        model.fit(train_X_seq_padded, train_y, batch_size=TRAIN_BATCH_SIZE, 
                  epochs=EPOCHS, callbacks=[checkpoint],verbose=0, 
                  validation_data=(val_X_seq_padded, val_y))
        
        model.load_weights(checkpoint_name)
        y_pred = model.predict(test_X_seq_padded,batch_size=TEST_BATCH_SIZE,verbose=0)
        
        y_pred = y_pred.argmax(1)
        y_pred = y_pred.astype(np.int8)
        
        accuracy = metrics.accuracy_score(test_y.argmax(1), y_pred)
        f1_score_micro = metrics.f1_score(test_y.argmax(1), y_pred, average='micro')
        f1_score_macro = metrics.f1_score(test_y.argmax(1), y_pred, average='macro')
        precision_macro = metrics.precision_score(test_y.argmax(1), y_pred, average='macro')
        recall_macro = metrics.recall_score(test_y.argmax(1), y_pred, average='macro')
        
        test_metrics.append({'n_exp':n_exp,'n_split':n_split,'epoch':15,
            'accuracy':accuracy,
            'f1_micro':f1_score_micro,
            'f1_macro':f1_score_macro,
            'precision_macro':precision_macro,
            'recall_macro':recall_macro,
                            })
        
        del model
        for _ in range(5): gc.collect()
        

        f = open("BILSTM_Results/scenario2_test_metrics.pkl","wb")
        pickle.dump(test_metrics,f)
        f.close()
        #print(test_metrics)

test_metrics = []


checkpoint_name = 'BILSTM_Scenario3.h5'
checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_name, verbose=0, 
                                             monitor='val_accuracy',save_best_only=True, mode='auto')
kf = StratifiedKFold(n_splits=5) # n-fold = 5
from tqdm.notebook import tqdm
for n_exp in tqdm(range(5)):
    best_losses = []
    best_metrics = []
    n_split=0
    kf = StratifiedKFold(n_splits=5) # n-fold = 5
    for train_idx,test_idx in tqdm(kf.split(X_scenario3,y_coded_scenario3),total=5):
        n_split+=1
        temp_X, temp_y = X_scenario3[train_idx], y_coded_scenario3[train_idx]
        train_X, val_X, train_y, val_y =  train_test_split(temp_X,temp_y, test_size=0.1, shuffle=True)
        test_X, test_y = X_scenario3[test_idx], y_coded_scenario3[test_idx]
        
        tokenizer = keras.preprocessing.text.Tokenizer(num_words = MAX_FEATURES, oov_token='<unk>', )
        tokenizer.fit_on_texts(train_X)
        
        train_X_seq = tokenizer.texts_to_sequences(train_X)
        val_X_seq = tokenizer.texts_to_sequences(val_X)
        test_X_seq = tokenizer.texts_to_sequences(test_X)
        
        train_X_seq_padded = keras.preprocessing.sequence.pad_sequences(train_X_seq,maxlen= MAX_LEN)
        val_X_seq_padded = keras.preprocessing.sequence.pad_sequences(val_X_seq,maxlen= MAX_LEN)
        test_X_seq_padded = keras.preprocessing.sequence.pad_sequences(test_X_seq,maxlen= MAX_LEN)
        
        emb_matrix = np.random.rand(MAX_FEATURES + 1,300)
        
        for i in range(1, MAX_FEATURES + 1):
#We get the word corresponding to index i
            word = tokenizer.index_word[i]
            if word in wv: #We ask if the word is in the word vector model
                emb_matrix[i] = wv[word] # We assign the value to the row to the word vector
        
        model = get_model(hidden_dim=128, output_dim=1, maxlen=MAX_LEN, 
                          max_features=MAX_FEATURES,emb_matrix=emb_matrix,last_activation='sigmoid')
        model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
        model.fit(train_X_seq_padded, train_y, batch_size=TRAIN_BATCH_SIZE, 
                  epochs=EPOCHS, callbacks=[checkpoint],verbose=0, 
                  validation_data=(val_X_seq_padded, val_y))
        
        model.load_weights(checkpoint_name)
        y_pred = model.predict(test_X_seq_padded,batch_size=TEST_BATCH_SIZE,verbose=0)
        
        y_pred = y_pred > 0.5
        y_pred = y_pred.astype(np.int8).reshape(-1)
        
        accuracy = metrics.accuracy_score(test_y, y_pred)
        f1_score_micro = metrics.f1_score(test_y, y_pred, average='micro')
        f1_score_macro = metrics.f1_score(test_y, y_pred, average='macro')
        precision_macro = metrics.precision_score(test_y, y_pred, average='macro')
        recall_macro = metrics.recall_score(test_y, y_pred, average='macro')
        
        test_metrics.append({'n_exp':n_exp,'n_split':n_split,'epoch':15,
            'accuracy':accuracy,
            'f1_micro':f1_score_micro,
            'f1_macro':f1_score_macro,
            'precision_macro':precision_macro,
            'recall_macro':recall_macro,
                            })
        
        del model
        for _ in range(5): gc.collect()
        

        f = open("BILSTM_Results/scenario3_test_metrics.pkl","wb")
        pickle.dump(test_metrics,f)
        f.close()
        #print(test_metrics)
#CNN
#Scenario 1
def get_model(hidden_dim, output_dim, maxlen, max_features, emb_matrix,last_activation = 'softmax'):
    inputs = keras.Input(shape=(maxlen,), dtype="int32")
# We transform each index into its corresponding vector of words
    # But this time, using the weights parameter, we initialize this vector layer
    # With the weights extracted from the pre-trained model
    # Besides, we define that this layer is not trainable, that is, the values ​​of the weights
    # Do not adjust in each iteration of training
    x = layers.Embedding(max_features + 1, 300,weights=[emb_matrix], trainable=False)(inputs)
    # Add a layer of LSTM
    #x = layers.Dropout(0.3)(x)

    # Conv1D + global max pooling
    x = layers.Conv1D(hidden_dim, 2, padding="valid", activation="relu", strides=2)(x)
    x = layers.Conv1D(hidden_dim, 2, padding="valid", activation="relu", strides=2)(x)

    x = layers.GlobalMaxPooling1D()(x)

# We add an intermediate layer to process the convolutions + pooling
    x = layers.Dense(int(hidden_dim/2), activation="tanh")(x)
    #x = layers.Dropout(0.3)(x)
    #x = layers.Dropout(0.3)(x)
    # We add the output layer, 1 output neuron because it is binary classification
    # In addition, we use the sigmoid activation function to return the probability
    outputs = layers.Dense(output_dim, activation=last_activation)(x)

    #We generate the model
    model = keras.Model(inputs, outputs)
    #model1.summary()
    return model
test_metrics = []


checkpoint_name = 'CNN_Scenario1.h5'
checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_name, verbose=0, 
                                             monitor='val_accuracy',save_best_only=True, mode='auto')
kf = StratifiedKFold(n_splits=5) # n-fold = 5
from tqdm.notebook import tqdm
for n_exp in tqdm(range(5)):
    best_losses = []
    best_metrics = []
    n_split=0
    kf = StratifiedKFold(n_splits=5) # n-fold = 5
    for train_idx,test_idx in tqdm(kf.split(X_scenario1,y_coded_scenario1.argmax(1)),total=5):
        n_split+=1
        temp_X, temp_y = X_scenario1[train_idx], y_coded_scenario1[train_idx]
        train_X, val_X, train_y, val_y =  train_test_split(temp_X,temp_y, test_size=0.1, shuffle=True)
        test_X, test_y = X_scenario1[test_idx], y_coded_scenario1[test_idx]
        
        tokenizer = keras.preprocessing.text.Tokenizer(num_words = MAX_FEATURES, oov_token='<unk>', )
        tokenizer.fit_on_texts(train_X)
        
        train_X_seq = tokenizer.texts_to_sequences(train_X)
        val_X_seq = tokenizer.texts_to_sequences(val_X)
        test_X_seq = tokenizer.texts_to_sequences(test_X)
        
        train_X_seq_padded = keras.preprocessing.sequence.pad_sequences(train_X_seq,maxlen= MAX_LEN)
        val_X_seq_padded = keras.preprocessing.sequence.pad_sequences(val_X_seq,maxlen= MAX_LEN)
        test_X_seq_padded = keras.preprocessing.sequence.pad_sequences(test_X_seq,maxlen= MAX_LEN)
        
        emb_matrix = np.random.rand(MAX_FEATURES + 1,300)
        
        for i in range(1, MAX_FEATURES + 1):
#We get the word corresponding to index i
            word = tokenizer.index_word[i]
            if word in wv: #We ask if the word is in the word vector model
                emb_matrix[i] = wv[word] # We assign the value to the row to the word vector
        
        model = get_model(hidden_dim=128, output_dim=3, maxlen=MAX_LEN, 
                          max_features=MAX_FEATURES,emb_matrix=emb_matrix,last_activation='softmax')
        model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
        model.fit(train_X_seq_padded, train_y, batch_size=TRAIN_BATCH_SIZE, 
                  epochs=EPOCHS, callbacks=[checkpoint],verbose=0, 
                  validation_data=(val_X_seq_padded, val_y))
        
        model.load_weights(checkpoint_name)
        y_pred = model.predict(test_X_seq_padded,batch_size=TEST_BATCH_SIZE,verbose=0)
        
        y_pred = y_pred.argmax(1)
        y_pred = y_pred.astype(np.int8)
        
        accuracy = metrics.accuracy_score(test_y.argmax(1), y_pred)
        f1_score_micro = metrics.f1_score(test_y.argmax(1), y_pred, average='micro')
        f1_score_macro = metrics.f1_score(test_y.argmax(1), y_pred, average='macro')
        precision_macro = metrics.precision_score(test_y.argmax(1), y_pred, average='macro')
        recall_macro = metrics.recall_score(test_y.argmax(1), y_pred, average='macro')
        
        test_metrics.append({'n_exp':n_exp,'n_split':n_split,'epoch':15,
            'accuracy':accuracy,
            'f1_micro':f1_score_micro,
            'f1_macro':f1_score_macro,
            'precision_macro':precision_macro,
            'recall_macro':recall_macro,
                            })
        
        del model
        for _ in range(5): gc.collect()
        

        f = open("CNN_Results/scenario1_test_metrics.pkl","wb")
        pickle.dump(test_metrics,f)
        f.close()
        #print(test_metrics)

#Scenario 2
        test_metrics = []


checkpoint_name = 'CNN_Scenario2.h5'
checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_name, verbose=0, 
                                             monitor='val_accuracy',save_best_only=True, mode='auto')

from tqdm.notebook import tqdm
for n_exp in tqdm(range(5)):
    best_losses = []
    best_metrics = []
    n_split=0
    kf = StratifiedKFold(n_splits=5) # n-fold = 5
    for train_idx,test_idx in tqdm(kf.split(X_scenario2,y_coded_scenario2.argmax(1)),total=5):
        n_split+=1
        temp_X, temp_y = X_scenario2[train_idx], y_coded_scenario2[train_idx]
        train_X, val_X, train_y, val_y =  train_test_split(temp_X,temp_y, test_size=0.1, shuffle=True)
        test_X, test_y = X_scenario2[test_idx], y_coded_scenario2[test_idx]
        
        tokenizer = keras.preprocessing.text.Tokenizer(num_words = MAX_FEATURES, oov_token='<unk>', )
        tokenizer.fit_on_texts(train_X)
        
        train_X_seq = tokenizer.texts_to_sequences(train_X)
        val_X_seq = tokenizer.texts_to_sequences(val_X)
        test_X_seq = tokenizer.texts_to_sequences(test_X)
        
        train_X_seq_padded = keras.preprocessing.sequence.pad_sequences(train_X_seq,maxlen= MAX_LEN)
        val_X_seq_padded = keras.preprocessing.sequence.pad_sequences(val_X_seq,maxlen= MAX_LEN)
        test_X_seq_padded = keras.preprocessing.sequence.pad_sequences(test_X_seq,maxlen= MAX_LEN)
        
        emb_matrix = np.random.rand(MAX_FEATURES + 1,300)
        
        for i in range(1, MAX_FEATURES + 1):
#We get the word corresponding to index i
            word = tokenizer.index_word[i]
            if word in wv: #We ask if the word is in the word vector model
                emb_matrix[i] = wv[word] # We assign the value to the row to the word vector
        
        model = get_model(hidden_dim=128, output_dim=3, maxlen=MAX_LEN, 
                          max_features=MAX_FEATURES,emb_matrix=emb_matrix,last_activation='softmax')
        model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
        model.fit(train_X_seq_padded, train_y, batch_size=TRAIN_BATCH_SIZE, 
                  epochs=EPOCHS, callbacks=[checkpoint],verbose=0, 
                  validation_data=(val_X_seq_padded, val_y))
        
        model.load_weights(checkpoint_name)
        y_pred = model.predict(test_X_seq_padded,batch_size=TEST_BATCH_SIZE,verbose=0)
        
        y_pred = y_pred.argmax(1)
        y_pred = y_pred.astype(np.int8)
        
        accuracy = metrics.accuracy_score(test_y.argmax(1), y_pred)
        f1_score_micro = metrics.f1_score(test_y.argmax(1), y_pred, average='micro')
        f1_score_macro = metrics.f1_score(test_y.argmax(1), y_pred, average='macro')
        precision_macro = metrics.precision_score(test_y.argmax(1), y_pred, average='macro')
        recall_macro = metrics.recall_score(test_y.argmax(1), y_pred, average='macro')
        
        test_metrics.append({'n_exp':n_exp,'n_split':n_split,'epoch':15,
            'accuracy':accuracy,
            'f1_micro':f1_score_micro,
            'f1_macro':f1_score_macro,
            'precision_macro':precision_macro,
            'recall_macro':recall_macro,
                            })
        
        del model
        for _ in range(5): gc.collect()
        

        f = open("CNN_Results/scenario2_test_metrics.pkl","wb")
        pickle.dump(test_metrics,f)
        f.close()
        #print(test_metrics)

#Scenario 3
        test_metrics = []


checkpoint_name = 'CNN_Scenario3.h5'
checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_name, verbose=0, 
                                             monitor='val_accuracy',save_best_only=True, mode='auto')
kf = StratifiedKFold(n_splits=5) # n-fold = 5
from tqdm.notebook import tqdm
for n_exp in tqdm(range(5)):
    best_losses = []
    best_metrics = []
    n_split=0
    kf = StratifiedKFold(n_splits=5) # n-fold = 5
    for train_idx,test_idx in tqdm(kf.split(X_scenario3,y_coded_scenario3),total=5):
        n_split+=1
        temp_X, temp_y = X_scenario3[train_idx], y_coded_scenario3[train_idx]
        train_X, val_X, train_y, val_y =  train_test_split(temp_X,temp_y, test_size=0.1, shuffle=True)
        test_X, test_y = X_scenario3[test_idx], y_coded_scenario3[test_idx]
        
        tokenizer = keras.preprocessing.text.Tokenizer(num_words = MAX_FEATURES, oov_token='<unk>', )
        tokenizer.fit_on_texts(train_X)
        
        train_X_seq = tokenizer.texts_to_sequences(train_X)
        val_X_seq = tokenizer.texts_to_sequences(val_X)
        test_X_seq = tokenizer.texts_to_sequences(test_X)
        
        train_X_seq_padded = keras.preprocessing.sequence.pad_sequences(train_X_seq,maxlen= MAX_LEN)
        val_X_seq_padded = keras.preprocessing.sequence.pad_sequences(val_X_seq,maxlen= MAX_LEN)
        test_X_seq_padded = keras.preprocessing.sequence.pad_sequences(test_X_seq,maxlen= MAX_LEN)
        
        emb_matrix = np.random.rand(MAX_FEATURES + 1,300)
        
        for i in range(1, MAX_FEATURES + 1):
#We get the word corresponding to index i
            word = tokenizer.index_word[i]
            if word in wv: #We ask if the word is in the word vector model
                emb_matrix[i] = wv[word] # We assign the value to the row to the word vector
        
        model = get_model(hidden_dim=128, output_dim=1, maxlen=MAX_LEN, 
                          max_features=MAX_FEATURES,emb_matrix=emb_matrix,last_activation='sigmoid')
        model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
        model.fit(train_X_seq_padded, train_y, batch_size=TRAIN_BATCH_SIZE, 
                  epochs=EPOCHS, callbacks=[checkpoint],verbose=0, 
                  validation_data=(val_X_seq_padded, val_y))
        
        model.load_weights(checkpoint_name)
        y_pred = model.predict(test_X_seq_padded,batch_size=TEST_BATCH_SIZE,verbose=0)
        
        y_pred = y_pred > 0.5
        y_pred = y_pred.astype(np.int8).reshape(-1)
        
        accuracy = metrics.accuracy_score(test_y, y_pred)
        f1_score_micro = metrics.f1_score(test_y, y_pred, average='micro')
        f1_score_macro = metrics.f1_score(test_y, y_pred, average='macro')
        precision_macro = metrics.precision_score(test_y, y_pred, average='macro')
        recall_macro = metrics.recall_score(test_y, y_pred, average='macro')
        
        test_metrics.append({'n_exp':n_exp,'n_split':n_split,'epoch':15,
            'accuracy':accuracy,
            'f1_micro':f1_score_micro,
            'f1_macro':f1_score_macro,
            'precision_macro':precision_macro,
            'recall_macro':recall_macro,
                            })
        
        del model
        for _ in range(5): gc.collect()
        

        f = open("CNN_Results/scenario3_test_metrics.pkl","wb")
        pickle.dump(test_metrics,f)
        f.close()
        #print(test_metrics)
#GRU
#Scenario 1

        test_metrics = []


checkpoint_name = 'GRU_Scenario1.h5'
checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_name, verbose=0, 
                                             monitor='val_accuracy',save_best_only=True, mode='auto')
kf = StratifiedKFold(n_splits=5) # n-fold = 5
from tqdm.notebook import tqdm
for n_exp in tqdm(range(5)):
    best_losses = []
    best_metrics = []
    n_split=0
    kf = StratifiedKFold(n_splits=5) # n-fold = 5
    for train_idx,test_idx in tqdm(kf.split(X_scenario1,y_coded_scenario1.argmax(1)),total=5):
        n_split+=1
        temp_X, temp_y = X_scenario1[train_idx], y_coded_scenario1[train_idx]
        train_X, val_X, train_y, val_y =  train_test_split(temp_X,temp_y, test_size=0.1, shuffle=True)
        test_X, test_y = X_scenario1[test_idx], y_coded_scenario1[test_idx]
        
        tokenizer = keras.preprocessing.text.Tokenizer(num_words = MAX_FEATURES, oov_token='<unk>', )
        tokenizer.fit_on_texts(train_X)
        
        train_X_seq = tokenizer.texts_to_sequences(train_X)
        val_X_seq = tokenizer.texts_to_sequences(val_X)
        test_X_seq = tokenizer.texts_to_sequences(test_X)
        
        train_X_seq_padded = keras.preprocessing.sequence.pad_sequences(train_X_seq,maxlen= MAX_LEN)
        val_X_seq_padded = keras.preprocessing.sequence.pad_sequences(val_X_seq,maxlen= MAX_LEN)
        test_X_seq_padded = keras.preprocessing.sequence.pad_sequences(test_X_seq,maxlen= MAX_LEN)
        
        emb_matrix = np.random.rand(MAX_FEATURES + 1,300)
        
        for i in range(1, MAX_FEATURES + 1):
#We get the word corresponding to index i
            word = tokenizer.index_word[i]
            if word in wv: #We ask if the word is in the word vector model
                emb_matrix[i] = wv[word] # We assign the value to the row to the word vector
        
        model = get_model(hidden_dim=128, output_dim=3, maxlen=MAX_LEN, 
                          max_features=MAX_FEATURES,emb_matrix=emb_matrix,last_activation='softmax')
        model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
        model.fit(train_X_seq_padded, train_y, batch_size=TRAIN_BATCH_SIZE, 
                  epochs=EPOCHS, callbacks=[checkpoint],verbose=0, 
                  validation_data=(val_X_seq_padded, val_y))
        
        model.load_weights(checkpoint_name)
        y_pred = model.predict(test_X_seq_padded,batch_size=TEST_BATCH_SIZE,verbose=0)
        
        y_pred = y_pred.argmax(1)
        y_pred = y_pred.astype(np.int8)
        
        accuracy = metrics.accuracy_score(test_y.argmax(1), y_pred)
        f1_score_micro = metrics.f1_score(test_y.argmax(1), y_pred, average='micro')
        f1_score_macro = metrics.f1_score(test_y.argmax(1), y_pred, average='macro')
        precision_macro = metrics.precision_score(test_y.argmax(1), y_pred, average='macro')
        recall_macro = metrics.recall_score(test_y.argmax(1), y_pred, average='macro')
        
        test_metrics.append({'n_exp':n_exp,'n_split':n_split,'epoch':15,
            'accuracy':accuracy,
            'f1_micro':f1_score_micro,
            'f1_macro':f1_score_macro,
            'precision_macro':precision_macro,
            'recall_macro':recall_macro,
                            })
        
        del model
        for _ in range(5): gc.collect()
        

        f = open("GRU_Results/scenario1_test_metrics.pkl","wb")
        pickle.dump(test_metrics,f)
        f.close()
        #print(test_metrics)

#Scenario 2

        test_metrics = []


checkpoint_name = 'GRU_Scenario2.h5'
checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_name, verbose=0, 
                                             monitor='val_accuracy',save_best_only=True, mode='auto')
kf = StratifiedKFold(n_splits=5) # n-fold = 5
from tqdm.notebook import tqdm
for n_exp in tqdm(range(5)):
    best_losses = []
    best_metrics = []
    n_split=0
    kf = StratifiedKFold(n_splits=5) # n-fold = 5
    for train_idx,test_idx in tqdm(kf.split(X_scenario2,y_coded_scenario2.argmax(1)),total=5):
        n_split+=1
        temp_X, temp_y = X_scenario2[train_idx], y_coded_scenario2[train_idx]
        train_X, val_X, train_y, val_y =  train_test_split(temp_X,temp_y, test_size=0.1, shuffle=True)
        test_X, test_y = X_scenario2[test_idx], y_coded_scenario2[test_idx]
        
        tokenizer = keras.preprocessing.text.Tokenizer(num_words = MAX_FEATURES, oov_token='<unk>', )
        tokenizer.fit_on_texts(train_X)
        
        train_X_seq = tokenizer.texts_to_sequences(train_X)
        val_X_seq = tokenizer.texts_to_sequences(val_X)
        test_X_seq = tokenizer.texts_to_sequences(test_X)
        
        train_X_seq_padded = keras.preprocessing.sequence.pad_sequences(train_X_seq,maxlen= MAX_LEN)
        val_X_seq_padded = keras.preprocessing.sequence.pad_sequences(val_X_seq,maxlen= MAX_LEN)
        test_X_seq_padded = keras.preprocessing.sequence.pad_sequences(test_X_seq,maxlen= MAX_LEN)
        
        emb_matrix = np.random.rand(MAX_FEATURES + 1,300)
        
        for i in range(1, MAX_FEATURES + 1):
#We get the word corresponding to index i
            word = tokenizer.index_word[i]
            if word in wv: #We ask if the word is in the word vector model
                emb_matrix[i] = wv[word] # We assign the value to the row to the word vector
        
        model = get_model(hidden_dim=128, output_dim=3, maxlen=MAX_LEN, 
                          max_features=MAX_FEATURES,emb_matrix=emb_matrix,last_activation='softmax')
        model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
        model.fit(train_X_seq_padded, train_y, batch_size=TRAIN_BATCH_SIZE, 
                  epochs=EPOCHS, callbacks=[checkpoint],verbose=0, 
                  validation_data=(val_X_seq_padded, val_y))
        
        model.load_weights(checkpoint_name)
        y_pred = model.predict(test_X_seq_padded,batch_size=TEST_BATCH_SIZE,verbose=0)
        
        y_pred = y_pred.argmax(1)
        y_pred = y_pred.astype(np.int8)
        
        accuracy = metrics.accuracy_score(test_y.argmax(1), y_pred)
        f1_score_micro = metrics.f1_score(test_y.argmax(1), y_pred, average='micro')
        f1_score_macro = metrics.f1_score(test_y.argmax(1), y_pred, average='macro')
        precision_macro = metrics.precision_score(test_y.argmax(1), y_pred, average='macro')
        recall_macro = metrics.recall_score(test_y.argmax(1), y_pred, average='macro')
        
        test_metrics.append({'n_exp':n_exp,'n_split':n_split,'epoch':15,
            'accuracy':accuracy,
            'f1_micro':f1_score_micro,
            'f1_macro':f1_score_macro,
            'precision_macro':precision_macro,
            'recall_macro':recall_macro,
                            })
        
        del model
        for _ in range(5): gc.collect()
        

        f = open("GRU_Results/scenario2_test_metrics.pkl","wb")
        pickle.dump(test_metrics,f)
        f.close()
        #print(test_metrics)

#Scenario 3
        test_metrics = []


checkpoint_name = 'GRU_Scenario3.h5'
checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_name, verbose=0, 
                                             monitor='val_accuracy',save_best_only=True, mode='auto')

from tqdm.notebook import tqdm
for n_exp in tqdm(range(5)):
    best_losses = []
    best_metrics = []
    n_split=0
    kf = StratifiedKFold(n_splits=5) # n-fold = 5
    for train_idx,test_idx in tqdm(kf.split(X_scenario3,y_coded_scenario3),total=5):
        n_split+=1
        temp_X, temp_y = X_scenario3[train_idx], y_coded_scenario3[train_idx]
        train_X, val_X, train_y, val_y =  train_test_split(temp_X,temp_y, test_size=0.1, shuffle=True)
        test_X, test_y = X_scenario3[test_idx], y_coded_scenario3[test_idx]
        
        tokenizer = keras.preprocessing.text.Tokenizer(num_words = MAX_FEATURES, oov_token='<unk>', )
        tokenizer.fit_on_texts(train_X)
        
        train_X_seq = tokenizer.texts_to_sequences(train_X)
        val_X_seq = tokenizer.texts_to_sequences(val_X)
        test_X_seq = tokenizer.texts_to_sequences(test_X)
        
        train_X_seq_padded = keras.preprocessing.sequence.pad_sequences(train_X_seq,maxlen= MAX_LEN)
        val_X_seq_padded = keras.preprocessing.sequence.pad_sequences(val_X_seq,maxlen= MAX_LEN)
        test_X_seq_padded = keras.preprocessing.sequence.pad_sequences(test_X_seq,maxlen= MAX_LEN)
        
        emb_matrix = np.random.rand(MAX_FEATURES + 1,300)
        
        for i in range(1, MAX_FEATURES + 1):
            #Obtenemos la palabra correspondiente al indice i
            word = tokenizer.index_word[i]
            if word in wv: #Preguntamos si la palabra esta en el modelo de vectores de palabra
                emb_matrix[i] = wv[word] # Asignamos el valor a la fila al vector de palabra
        
        model = get_model(hidden_dim=128, output_dim=1, maxlen=MAX_LEN, 
                          max_features=MAX_FEATURES,emb_matrix=emb_matrix,last_activation='sigmoid')
        model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
        model.fit(train_X_seq_padded, train_y, batch_size=TRAIN_BATCH_SIZE, 
                  epochs=EPOCHS, callbacks=[checkpoint],verbose=0, 
                  validation_data=(val_X_seq_padded, val_y))
        
        model.load_weights(checkpoint_name)
        y_pred = model.predict(test_X_seq_padded,batch_size=TEST_BATCH_SIZE,verbose=0)
        
        y_pred = y_pred > 0.5
        y_pred = y_pred.astype(np.int8).reshape(-1)
        
        accuracy = metrics.accuracy_score(test_y, y_pred)
        f1_score_micro = metrics.f1_score(test_y, y_pred, average='micro')
        f1_score_macro = metrics.f1_score(test_y, y_pred, average='macro')
        precision_macro = metrics.precision_score(test_y, y_pred, average='macro')
        recall_macro = metrics.recall_score(test_y, y_pred, average='macro')
        
        test_metrics.append({'n_exp':n_exp,'n_split':n_split,'epoch':15,
            'accuracy':accuracy,
            'f1_micro':f1_score_micro,
            'f1_macro':f1_score_macro,
            'precision_macro':precision_macro,
            'recall_macro':recall_macro,
                            })
        
        del model
        for _ in range(5): gc.collect()
        

        f = open("GRU_Results/scenario3_test_metrics.pkl","wb")
        pickle.dump(test_metrics,f)
        f.close()
        #print(test_metrics)
#LSTM
#Scenario 1

        from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score


def get_model(hidden_dim, output_dim, maxlen, max_features, emb_matrix,last_activation = 'softmax'):
    #Input layer, which will receive the interos arrays (vocabulary indexes)
    inputs = keras.Input(shape=(maxlen,), dtype="int32")
    # We transform each index into its corresponding vector of words
    # But this time, using the weights parameter, we initialize this vector layer
    # With the weights extracted from the pre-trained model
    # Besides, we define that this layer is not trainable, that is, the values ​​of the weights
    # Do not adjust in each iteration of training
    x = layers.Embedding(max_features + 1, 300,weights=[emb_matrix], trainable=False)(inputs)
    # Add a layer of LSTM
    x = layers.LSTM(hidden_dim)(x)
    #x = layers.Dropout(0.3)(x)
    # We add the output layer, 1 output neuron because it is binary classification
    # In addition, we use the sigmoid activation function to return the probability
    outputs = layers.Dense(output_dim, activation=last_activation)(x)

#We generate the model
    model = keras.Model(inputs, outputs)
    #model1.summary()
    return model

test_metrics = []


checkpoint_name = 'LSTM_Scenario1.h5'
checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_name, verbose=0, 
                                             monitor='val_accuracy',save_best_only=True, mode='auto')
kf = StratifiedKFold(n_splits=5) # n-fold = 5
from tqdm.notebook import tqdm
for n_exp in tqdm(range(5)):
    best_losses = []
    best_metrics = []
    n_split=0
    kf = StratifiedKFold(n_splits=5) # n-fold = 5
    for train_idx,test_idx in tqdm(kf.split(X_scenario1,y_coded_scenario1.argmax(1)),total=5):
        n_split+=1
        temp_X, temp_y = X_scenario1[train_idx], y_coded_scenario1[train_idx]
        train_X, val_X, train_y, val_y =  train_test_split(temp_X,temp_y, test_size=0.1, shuffle=True)
        test_X, test_y = X_scenario1[test_idx], y_coded_scenario1[test_idx]
        
        tokenizer = keras.preprocessing.text.Tokenizer(num_words = MAX_FEATURES, oov_token='<unk>', )
        tokenizer.fit_on_texts(train_X)
        
        train_X_seq = tokenizer.texts_to_sequences(train_X)
        val_X_seq = tokenizer.texts_to_sequences(val_X)
        test_X_seq = tokenizer.texts_to_sequences(test_X)
        
        train_X_seq_padded = keras.preprocessing.sequence.pad_sequences(train_X_seq,maxlen= MAX_LEN)
        val_X_seq_padded = keras.preprocessing.sequence.pad_sequences(val_X_seq,maxlen= MAX_LEN)
        test_X_seq_padded = keras.preprocessing.sequence.pad_sequences(test_X_seq,maxlen= MAX_LEN)
        
        emb_matrix = np.random.rand(MAX_FEATURES + 1,300)
        
        for i in range(1, MAX_FEATURES + 1):
#We get the word corresponding to index i
            word = tokenizer.index_word[i]
            if word in wv: #We ask if the word is in the word vector model
                emb_matrix[i] = wv[word] # We assign the value to the row to the word vector
        
        model = get_model(hidden_dim=128, output_dim=3, maxlen=MAX_LEN, 
                          max_features=MAX_FEATURES,emb_matrix=emb_matrix,last_activation='softmax')
        model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
        model.fit(train_X_seq_padded, train_y, batch_size=TRAIN_BATCH_SIZE, 
                  epochs=EPOCHS, callbacks=[checkpoint],verbose=0, 
                  validation_data=(val_X_seq_padded, val_y))
        
        model.load_weights(checkpoint_name)
        y_pred = model.predict(test_X_seq_padded,batch_size=TEST_BATCH_SIZE,verbose=0)
        
        y_pred = y_pred.argmax(1)
        y_pred = y_pred.astype(np.int8)
        
        accuracy = metrics.accuracy_score(test_y.argmax(1), y_pred)
        f1_score_micro = metrics.f1_score(test_y.argmax(1), y_pred, average='micro')
        f1_score_macro = metrics.f1_score(test_y.argmax(1), y_pred, average='macro')
        precision_macro = metrics.precision_score(test_y.argmax(1), y_pred, average='macro')
        recall_macro = metrics.recall_score(test_y.argmax(1), y_pred, average='macro')
        
        test_metrics.append({'n_exp':n_exp,'n_split':n_split,'epoch':15,
            'accuracy':accuracy,
            'f1_micro':f1_score_micro,
            'f1_macro':f1_score_macro,
            'precision_macro':precision_macro,
            'recall_macro':recall_macro,
                            })
        
        del model
        for _ in range(5): gc.collect()
        

        f = open("LSTM_Results/scenario1_test_metrics.pkl","wb")
        pickle.dump(test_metrics,f)
        f.close()
        #print(test_metrics)

#Scenario 2
test_metrics = []


checkpoint_name = 'LSTM_Scenario2.h5'
checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_name, verbose=0, 
                                             monitor='val_accuracy',save_best_only=True, mode='auto')
kf = StratifiedKFold(n_splits=5) # n-fold = 5
from tqdm.notebook import tqdm
for n_exp in tqdm(range(5)):
    best_losses = []
    best_metrics = []
    n_split=0
    kf = StratifiedKFold(n_splits=5) # n-fold = 5
    for train_idx,test_idx in tqdm(kf.split(X_scenario2,y_coded_scenario2.argmax(1)),total=5):
        n_split+=1
        temp_X, temp_y = X_scenario2[train_idx], y_coded_scenario2[train_idx]
        train_X, val_X, train_y, val_y =  train_test_split(temp_X,temp_y, test_size=0.1, shuffle=True)
        test_X, test_y = X_scenario2[test_idx], y_coded_scenario2[test_idx]
        
        tokenizer = keras.preprocessing.text.Tokenizer(num_words = MAX_FEATURES, oov_token='<unk>', )
        tokenizer.fit_on_texts(train_X)
        
        train_X_seq = tokenizer.texts_to_sequences(train_X)
        val_X_seq = tokenizer.texts_to_sequences(val_X)
        test_X_seq = tokenizer.texts_to_sequences(test_X)
        
        train_X_seq_padded = keras.preprocessing.sequence.pad_sequences(train_X_seq,maxlen= MAX_LEN)
        val_X_seq_padded = keras.preprocessing.sequence.pad_sequences(val_X_seq,maxlen= MAX_LEN)
        test_X_seq_padded = keras.preprocessing.sequence.pad_sequences(test_X_seq,maxlen= MAX_LEN)
        
        emb_matrix = np.random.rand(MAX_FEATURES + 1,300)
        
        for i in range(1, MAX_FEATURES + 1):
#We get the word corresponding to index i
            word = tokenizer.index_word[i]
            if word in wv: #We ask if the word is in the word vector model
                emb_matrix[i] = wv[word] # We assign the value to the row to the word vector
        
        model = get_model(hidden_dim=128, output_dim=3, maxlen=MAX_LEN, 
                          max_features=MAX_FEATURES,emb_matrix=emb_matrix,last_activation='softmax')
        model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
        model.fit(train_X_seq_padded, train_y, batch_size=TRAIN_BATCH_SIZE, 
                  epochs=EPOCHS, callbacks=[checkpoint],verbose=0, 
                  validation_data=(val_X_seq_padded, val_y))
        
        model.load_weights(checkpoint_name)
        y_pred = model.predict(test_X_seq_padded,batch_size=TEST_BATCH_SIZE,verbose=0)
        
        y_pred = y_pred.argmax(1)
        y_pred = y_pred.astype(np.int8)
        
        accuracy = metrics.accuracy_score(test_y.argmax(1), y_pred)
        f1_score_micro = metrics.f1_score(test_y.argmax(1), y_pred, average='micro')
        f1_score_macro = metrics.f1_score(test_y.argmax(1), y_pred, average='macro')
        precision_macro = metrics.precision_score(test_y.argmax(1), y_pred, average='macro')
        recall_macro = metrics.recall_score(test_y.argmax(1), y_pred, average='macro')
        
        test_metrics.append({'n_exp':n_exp,'n_split':n_split,'epoch':15,
            'accuracy':accuracy,
            'f1_micro':f1_score_micro,
            'f1_macro':f1_score_macro,
            'precision_macro':precision_macro,
            'recall_macro':recall_macro,
                            })
        
        del model
        for _ in range(5): gc.collect()
        

        f = open("LSTM_Results/scenario2_test_metrics.pkl","wb")
        pickle.dump(test_metrics,f)
        f.close()
        #print(test_metrics)

#Scenario 3
        test_metrics = []


checkpoint_name = 'LSTM_Scenario3.h5'
checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_name, verbose=0, 
                                             monitor='val_accuracy',save_best_only=True, mode='auto')
kf = StratifiedKFold(n_splits=5) # n-fold = 5
from tqdm.notebook import tqdm
for n_exp in tqdm(range(5)):
    best_losses = []
    best_metrics = []
    n_split=0
    kf = StratifiedKFold(n_splits=5) # n-fold = 5
    for train_idx,test_idx in tqdm(kf.split(X_scenario3,y_coded_scenario3),total=5):
        n_split+=1
        temp_X, temp_y = X_scenario3[train_idx], y_coded_scenario3[train_idx]
        train_X, val_X, train_y, val_y =  train_test_split(temp_X,temp_y, test_size=0.1, shuffle=True)
        test_X, test_y = X_scenario3[test_idx], y_coded_scenario3[test_idx]
        
        tokenizer = keras.preprocessing.text.Tokenizer(num_words = MAX_FEATURES, oov_token='<unk>', )
        tokenizer.fit_on_texts(train_X)
        
        train_X_seq = tokenizer.texts_to_sequences(train_X)
        val_X_seq = tokenizer.texts_to_sequences(val_X)
        test_X_seq = tokenizer.texts_to_sequences(test_X)
        
        train_X_seq_padded = keras.preprocessing.sequence.pad_sequences(train_X_seq,maxlen= MAX_LEN)
        val_X_seq_padded = keras.preprocessing.sequence.pad_sequences(val_X_seq,maxlen= MAX_LEN)
        test_X_seq_padded = keras.preprocessing.sequence.pad_sequences(test_X_seq,maxlen= MAX_LEN)
        
        emb_matrix = np.random.rand(MAX_FEATURES + 1,300)
        
        for i in range(1, MAX_FEATURES + 1):
           
            word = tokenizer.index_word[i]
            if word in wv: 
                emb_matrix[i] = wv[word] #
        
        model = get_model(hidden_dim=128, output_dim=1, maxlen=MAX_LEN, 
                          max_features=MAX_FEATURES,emb_matrix=emb_matrix,last_activation='sigmoid')
        model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
        model.fit(train_X_seq_padded, train_y, batch_size=TRAIN_BATCH_SIZE, 
                  epochs=EPOCHS, callbacks=[checkpoint],verbose=0, 
                  validation_data=(val_X_seq_padded, val_y))
        
        model.load_weights(checkpoint_name)
        y_pred = model.predict(test_X_seq_padded,batch_size=TEST_BATCH_SIZE,verbose=0)
        
        y_pred = y_pred > 0.5
        y_pred = y_pred.astype(np.int8).reshape(-1)
        
        accuracy = metrics.accuracy_score(test_y, y_pred)
        f1_score_micro = metrics.f1_score(test_y, y_pred, average='micro')
        f1_score_macro = metrics.f1_score(test_y, y_pred, average='macro')
        precision_macro = metrics.precision_score(test_y, y_pred, average='macro')
        recall_macro = metrics.recall_score(test_y, y_pred, average='macro')
        
        test_metrics.append({'n_exp':n_exp,'n_split':n_split,'epoch':15,
            'accuracy':accuracy,
            'f1_micro':f1_score_micro,
            'f1_macro':f1_score_macro,
            'precision_macro':precision_macro,
            'recall_macro':recall_macro,
                            })
        
        del model
        for _ in range(5): gc.collect()
        

        f = open("LSTM_Results/scenario3_test_metrics.pkl","wb")
        pickle.dump(test_metrics,f)
        f.close()
        #print(test_metrics)
