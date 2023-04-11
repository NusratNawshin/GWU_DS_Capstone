from flask import Flask, render_template, request, jsonify
# from keras.models import load_model
from keras.preprocessing import image
from single_image_captioning import result_caption
from train_and_validation import ImageCaptionModel,PositionalEncoding,DatasetLoader
import pyttsx3


app = Flask(__name__)

# dic = {0 : 'Cat', 1 : 'Dog'}

# model = load_model('model.h5')
#
# model.make_predict_function()
#
def predict_label(img_path):
	i = image.load_img(img_path, target_size=(100,100))
	i = image.img_to_array(i)/255.0
	i = i.reshape(1, 100,100,3)
	# p = model.predict_classes(i)
	# return dic[p[0]]
	# result_caption(img_path)
	return " "


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

# @app.route("/about")
# def about_page():
# 	return "GWU Capstone Project: Image Captionn Generator"

@app.route("/txt2speech", methods = ['GET', 'POST'])
def text_to_speech():
	text=request.form.get('caption')
	image=request.form.get('image_path')
	print("Hello this is me")
	print(text)
	print(image)
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