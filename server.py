import os
from flask import Flask, render_template, request
from werkzeug import secure_filename
import pandas as pd

app = Flask(__name__)
APP__ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/upload', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
		target = os.path.join(APP__ROOT, 'uploaded/')
		if not os.path.isdir(target):
			os.mkdir(target)
		f = request.files['file']
		destination = "/".join([target, 'input.csv'])
		f.save(destination)
		return 'file uploaded successfully'
count = 0

#@app.route("/tables")
#def show_tables():
   # data = pd.read_csv('output.csv')
    #data.set_index(['Name'], inplace=True)
    #data.index.name=None
    #Price = data.loc[data.test_id== count]
    #males = data.loc[data.Gender=='m']
    
    #return render_template('view.html',tables=[Price.to_html],
    #titles = ['S.no', 'Price'])
@app.route('/dataset')    
def another_page():    
  table = pd.DataFrame.from_csv("output.csv")
  return render_template("view.html", data=table.to_html)


if __name__ == '__main__':
	app.run(debug=True)
