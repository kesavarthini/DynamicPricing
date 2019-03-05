# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 10:40:02 2019

@author: Swarnalatha S
"""
import requests
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plotter
import matplotlib
import cgi, cgitb 
import io
import base64
import random
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from lxml import html
from lxml.html import parse
from bs4 import BeautifulSoup

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
import re
from sklearn import preprocessing
from sklearn import ensemble

print(check_output(["ls", "/home/swarnalatha/Desktop/DynamicPricing/templates/"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

train = pd.read_csv("/home/swarnalatha/Desktop/DynamicPricing/70_train.csv", sep=',')
test = pd.read_csv("/home/swarnalatha/Desktop/DynamicPricing/30_test.csv", sep=',')

train['item_description'] = train['item_description'].astype(str)
test['item_description'] = test['item_description'].astype(str)

test["name"] = test["name"].astype(str)
train["name"] = train["name"].astype(str)

train['des_len'] = train['item_description'].apply(lambda x: len(x))
test['des_len'] = test['item_description'].apply(lambda x: len(x))

# words in description
train['word_count'] = train['item_description'].apply(lambda x: len(x.split()))
test['word_count'] = test['item_description'].apply(lambda x: len(x.split()))
# men len of words in description inversed and scaled
train['mean_des'] = train['item_description'].apply(lambda x: float(len(x.split())) / len(x))  * 10
test['mean_des'] = test['item_description'].apply(lambda x: float(len(x.split())) / len(x)) * 10 
# length of name
train['name_len'] = train['name'].apply(lambda x: len(x))
test['name_len'] = test['name'].apply(lambda x: len(x))
# words in name
train['word_name'] = train['name'].apply(lambda x: len(x.split()))
test['word_name'] = test['name'].apply(lambda x: len(x.split()))
# mean len of words in name inversed and scaled
train['mean_name'] = train['name'].apply(lambda x: float(len(x.split())) / len(x))  * 10
test['mean_name'] = test['name'].apply(lambda x: float(len(x.split())) / len(x)) * 10 
train.head()
train.isnull().sum()
train['category_name'].value_counts()
train['category_name'].fillna('ppp,ppp,ppp', inplace=True)
test['category_name'].fillna('ppp,ppp,ppp', inplace=True)
#train[ train['category_name'].str.contains('Electronics') ]['price'].mean()
train['elec'] = train['category_name'].apply(lambda x : int('shoes' in x.lower()))
test['elec'] = test['category_name'].apply(lambda x : int('shoes' in x.lower()))
train['category_name'].value_counts()
train['brand_name'].fillna('ttttttt', inplace=True)
test['brand_name'].fillna('ttttttt', inplace=True)
train.isnull().sum()
test.isnull().sum()
# types of category
#train['cat_len'] = train['category_name'].apply(lambda x: len( x.split('/')))
#test['cat_len'] = test['category_name'].apply(lambda x: len( x.split('/')))
#length of category words
train['cat_lennn'] = train['category_name'].apply(lambda x: len(x))
test['cat_lennn'] = test['category_name'].apply(lambda x: len(x))
def was_priced(x):
    return int('[rm]' in x)
               
train['rm'] = train['item_description'].apply( lambda x : was_priced(x))
test['rm'] = test['item_description'].apply( lambda x : was_priced(x))
               
train['was_described'] = 1
test['was_described'] = 1

train.loc[ train['item_description'] == 'No description yet','was_described'] = 0
test.loc[ test['item_description'] == 'No description yet','was_described'] = 0
# description containes 'new' word
train['new'] = train['item_description'].apply(lambda x : int('new' in x.lower()))
test['new'] = test['item_description'].apply(lambda x : int('new' in x.lower()))
# splitting subcategories of category_name
train_cat = pd.DataFrame(train.category_name.str.split(',',2).tolist(),
                                   columns = ['sub1','sub2', 'sub3'])
train['sub1'] = train_cat['sub1']
train['sub2'] = train_cat['sub2']
train['sub3'] = train_cat['sub3']

test_cat = pd.DataFrame(test.category_name.str.split(',',2).tolist(),
                                   columns = ['sub1','sub2', 'sub3'])

test['sub1'] = test_cat['sub1']
test['sub2'] = test_cat['sub2']
test['sub3'] = test_cat['sub3']

print train.head()

train['clothing'] = train['category_name'].apply(lambda x : int('clothing' in x.lower()))
test['clothing'] = test['category_name'].apply(lambda x : int('clothing' in x.lower()))
train['shoes'] = train['category_name'].apply(lambda x : int('shoes' in x.lower()))
test['shoes'] = test['category_name'].apply(lambda x : int('shoes' in x.lower()))
train['int_desc'] = train['item_description'].apply(lambda x : int(bool(re.search(r'\d',x))))
test['int_desc'] = test['item_description'].apply(lambda x : int(bool(re.search(r'\d',x))))
# integer was present in name
train['int_name'] = train['name'].apply(lambda x : int(bool(re.search(r'\d',x))))
test['int_name'] = test['name'].apply(lambda x : int(bool(re.search(r'\d',x))))
# word condition was present in description
train['cond'] = train['item_description'].apply(lambda x : int('condition' in x.lower()))
test['cond'] = test['item_description'].apply(lambda x : int('condition' in x.lower()))
 
print train['category_name'].value_counts()
# converting price to log scale
positive = train['price'].values > 0
negative = train['price'].values < 0
train['price'] = np.piecewise(train['price'], (positive, negative), (np.log, lambda x: -np.log(-x)))

features = ['int_name','cond','int_desc', 'new', 'was_described', 'shoes', 'rm', 'item_condition_id','cat_lennn',  'brand_name', 'shipping', 'des_len', 'name_len','mean_des', 'word_count', 'mean_name', 'word_name', 'sub1', 'sub2', 'clothing', 'elec', 'category_name']

data = train[features]
data_sub = test[features]

y = train['price']
print data_sub.head()

#label encoding
le = preprocessing.LabelEncoder()

frames = [ data, data_sub ]
xx = pd.concat(frames)


l = [ 'brand_name', 'sub1', 'sub2', 'category_name']
for x in l :
    le.fit(xx[x])
    data[x] = le.transform(data[x])
    data_sub[x] = le.transform(data_sub[x])

for i in range(0,len(features)):
	print type(data[features[i]][0])

data.head()

clf =  ensemble.GradientBoostingRegressor( learning_rate = 0.7, n_estimators=700, max_depth = 3,warm_start = True, verbose=1, random_state=45, max_features = 0.8)
clf.fit(data, y)
# predicting and saving to output file
predicted = clf.predict(data_sub) 

print(features)
print( clf.feature_importances_)
out = pd.DataFrame()
#sample = pd.read_csv('../input/sample_submission.csv')
#out['test_id'] = test['test_id']
out['name']=test['name']
out['price'] = predicted
out['price'] = np.exp(out['price'])
out.head()
out.to_csv("output.csv",index=False)

#===================================================

import os
import logging
from flask import Flask, render_template, url_for, redirect, request, Response, jsonify
app = Flask(__name__)
APP__ROOT = os.path.dirname(os.path.abspath(__file__))

option = None

@app.route("/")
@app.route("/home")
def home():
	return render_template('index.html')

@app.route('/upload', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
		target = os.path.join(APP__ROOT, 'uploaded/')
		if not os.path.isdir(target):
			os.mkdir(target)
		f = request.files['file']
		option = request.form.get("category")
		app.logger.info(option)
		destination = "/".join([target, 'input.csv'])
		f.save(destination)
		return ("File Uploaded Successfully", 200)


@app.route('/dataset')    
def another_page():    
  table = pd.DataFrame.from_csv("output.csv")
  return render_template("view.html", data=table.to_html())


if __name__ == '__main__':
	app.run(debug=True)
