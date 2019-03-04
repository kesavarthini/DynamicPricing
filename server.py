import os
from flask import Flask, render_template, request
from werkzeug import secure_filename

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

if __name__ == '__main__':
	app.run(debug=True)