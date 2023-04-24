from flask import Flask, render_template, request, jsonify
# from keras.models import load_model
from keras.preprocessing import image
from single_image_captioning import result_caption
from train_and_validation import ImageCaptionModel,PositionalEncoding,DatasetLoader
import pyttsx3


app = Flask(__name__)

#
def predict_label(img_path):
	i = image.load_img(img_path, target_size=(100,100))
	i = image.img_to_array(i)/255.0
	i = i.reshape(1, 100,100,3)
	return " "


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

# @app.route("/about")
# def about_page():
# 	return "GWU Spring 2023 Capstone Project: Image Caption Generator"

@app.route("/txt2speech", methods = ['GET', 'POST'])
def text_to_speech():
	text=request.form.get('caption')
	image=request.form.get('image_path')
	# print(text)
	# print(image)
	engine = pyttsx3.init()
	engine.setProperty('rate', 110)
	engine.setProperty('volume', 0.8)
	voices = engine.getProperty('voices')
	engine.setProperty('voice', voices[1].id)
	engine.say(text)
	engine.runAndWait()
	return jsonify(status="Success")

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		if(request.files['my_image'].filename != ''):
			img = request.files['my_image']

			img_path = "static/" + img.filename
			img.save(img_path)
			#
			# p = predict_label(img_path)
			p = result_caption(img_path)
			# if(p != ""):

			return render_template("index.html", prediction = p, img_path = img_path,flag='read')
		else:
			return render_template("index.html")

	else:
		return render_template("index.html")



if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)