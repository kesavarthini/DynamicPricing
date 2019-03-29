DYNAMIC PRICING OF PRODUCTS BASED ON VISUAL QUALITY AND OTHER GENERIC E-COMMERCE FACTORS

README.txt

PREREQUISITES
--------------
Knowledge of the various particulars about the products to be priced and basic knowledge of Python and its packages

TO BE INSTALLED
.-----------------
-Python 3.0
-Flask
-Image
-Numpy
-Pandas
-SCIKIT Learn

LIBRARIES TO BE INSTALLED
-----------------------------
import requests
import time
import pandas 
import numpy 
import matplotlib.pyplot as plotter
import matplotlib
import cgi, cgitb 
import io
import base64
import random
from lxml import html
from lxml.html import parse
from bs4 import BeautifulSoup
from sklearn import preprocessing
from sklearn import ensemble
from itertools import izip
import Image


RUN PROCEDURE
------------------
-Download dataset of Product Details
-Change path name of dataset in dy.py accordingly
-Change path name of dataset accordingly in required training model 
-Run required training model 
-Use the saved trained model for predicting the optimal prices of products using Gradient boosting Regressor function

FOR EXECUTION
-----------------
-Run dy.py
-In web browser load the webpage "localhost:5000"
-Input the details of the product to be priced i.e : test_id, name, brand_name, category_name, item_description, item_condition_id, sale, img_score, supply_demand, shipping, image_url
- Run img_scraper.py to fetch the images from the URLs provided.
- Run img_cmp.py to compare the images and write them in the .csv file.
-Output is written into output.csv file
- To view the output, in web browser load the webpage "localhost:5000/dataset" to view the optimal prices given to products. 
