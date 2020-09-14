from flask import Flask, flash, request, redirect, url_for, render_template
import jsonify
import imageio
import pickle
import numpy as np
import sklearn
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.models import load_model
from numpy.linalg import norm
import pickle as pk
from PIL import Image
import cv2
import os

app = Flask(__name__)



# If you have Caltech data set uncomment below code

# path = 'static/Caltech101'
# if not os.path.isdir(path) :
# 	os.system("wget https://ndownloader.figshare.com/files/12855005")
# 	os.system('mv 12855005 files.rar')
# 	os.system('unrar x files.rar')
# 	os.system('rm files.rar')
# 	os.system('mkdir static')
# 	os.system('mv Caltech101')
# 	os.system('mv Caltech101 static')


# If you don't have resnet weights comment below line
model = load_model('resnet.h5')

# If you don't have resnet weights uncomment below line
# model = ResNet50(include_top=False, input_shape=(224, 224, 3), pooling='max')


filenames = pk.load(open('caltech101-filenames.pickle', 'rb'))
features = pk.load(open('caltech101-resnet50-features.pickle', 'rb'))

neighbours = NearestNeighbors().fit(features)

def extract_features_numpy(img):
	img = img[np.newaxis, :]
	preprocess_img = preprocess_input(img)
	features = model.predict(preprocess_img)
	features = features.flatten()
	features = features / norm(features)
	return features

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/search", methods=['POST'])
def search():
	if request.method == 'POST':
		# image = request.form['image'] # To load image with url
		image = request.files['image']
		no_of_img = int(request.form['quantity'])
		try:
			img = imageio.imread(image)
			l = []
			img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
			img = np.asarray(img)
			features = extract_features_numpy(img)
			features = np.expand_dims(features, axis=0)
			distances, indices = neighbours.kneighbors(features, n_neighbors=no_of_img)
			for i in indices[0]:
				l.append("/static/"+filenames[i])
		
			return render_template('index.html', filename=l)
		except :
					return render_template('index.html', return_code="Failed to process image")

@app.route('/Caltech101/<filename>')
def display_image(filename):
	for i in filename:
		print('display_image filename: ' + i)
	return redirect(url_for('static', filename=""+filename), code=301)

if __name__=="__main__":
	app.run(debug=True, host='0.0.0.0', port=8080)
