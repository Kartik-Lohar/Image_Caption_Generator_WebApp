import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from flask import Flask,render_template,request,session
from keras.models import Model,load_model
from keras.layers import LSTM,Dense,Embedding,Dropout
from keras.applications.xception import Xception
from PIL import Image
from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from pickle import load,dump
from werkzeug.utils import secure_filename

upload_folder = os.path.join("static")
app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = upload_folder

app.secret_key = "ima_kartik_generator"

@app.route("/")
def web_homepage():
    return render_template("index.html")


@app.route("/predictimage",methods = ["POST"])
def textprediction():
    if request.method== "POST":
        img = request.files["image"]
        img_filename = secure_filename(img.filename)
        img.save(os.path.join(upload_folder,img_filename))
        img_path = os.path.join(upload_folder,img_filename)

        pred_desc = test_descriptions(img_path)[0]

        print(pred_desc)
        
        pred_desc = " ".join(pred_desc.split()[1:-1]).capitalize()

        return render_template("index1.html",img = img_path, desc = pred_desc)    

def extract_feature(filename,model):
  """FUNCTION:
         extract_features:
                 Arguements: path of the image, xception model
                 returns: features of that image
  """

  image = Image.open(filename) #opening the image and saving it state

  image = image.resize((299,299)) # resizing the image into 299*299

  image = np.array(image) # converting the image values into numpy array

  # if image has four channels then we reduce it to 3
  if image.shape[2]==4:
    image = image[..., :3] # taking only 3 channels of the image(Example:- rgb = red,green,blue)

  image = np.expand_dims(image,axis = 0) # expanding or adding one more dimension

  image = image/127.5  # reducing the image pixels values

  image = image - 1.0 # reducing the image pixels values

  feature = model.predict(image) # predicting the features of the image using xception model of keras

  return feature #returning the features of the image

def word_for_id(integer,tokenizer):
  """FUNCTION:
       word_for_id:
             Arguements: integer index value of the word, tokenizer object which have all the words of our vocabulary
             returns: the word according to its index
  """
  for word,index in tokenizer.word_index.items(): # iterating through dictionary of words as keys and index value as values 
    
    if  index == integer: #searching for that index which is predicted by our model in vocabulary 
      return word #returning the word which index matches with our predicted index

  return None # returning None if that word is not present in our vocabulary

def generate_desc(model,tokenizer,photo,max_length):

  in_text = "start"  # inidicator that the description is start
  for i in range(max_length): #iterating upto maximum length of the caption

    sequence = tokenizer.texts_to_sequences([in_text])[0] #converting the text values into integers of sequences

    sequence = pad_sequences([sequence],maxlen = max_length) # padding the sequence to maxmin length of a caption

    pred = model.predict([photo,sequence],verbose = 0) # predicting the word of the description for that image

    pred =np.argmax(pred) #finding the index of the maximum value into the array

    word = word_for_id(pred,tokenizer) #calling the function for getting the word according to this index value in our vocabulary

    if word is None: # if word is not present in vocbulary
      break
    in_text += " " + word # adding that word in the in_text
    if word == "end": # it means we have reached the end of description
      break
  return in_text # returning the final description of that image


def test_descriptions(test_set):
  """FUNCTION:
        test_descriptions:
              Arguements: list of images name 
              returns: list of predicted descriptions
  """
  pred_desc = []  #initializing the list

  #loading the last trained model after completing each epoch
  model = load_model(r"C:\Users\Kartik\Desktop\Image_caption_generator\model_1\model_9.h5")

  # getting the xception model of keras library for getting feature of the given image
  xception_model = Xception(include_top = False,pooling = "avg")
  

  #calling function for extracting the feature of given image
  photo = extract_feature(test_set,xception_model)

  # load tokenizer
  tokenizer = load(open(r"C:\Users\Kartik\Desktop\Image_caption_generator\tokenizer.p","rb"))

  # max length hard code just for running webapplication
  max_length = 20

  # getting the caption of the given image whcih describes the image (or a text descriptions of a image)
  description = generate_desc(model,tokenizer,photo,max_length)  

    # appending the description of the image
  pred_desc.append(description)

  return pred_desc  #returning the list of predicted descriptions


# with open("/content/Flickr_8k.testImages.txt","r") as f:

#   test_set = f.read() # getting the string of all the images name in test set  

# test_set = test_set.split("\n")[:-1] # splitting the images 

# pred_description = test_descriptions(test_set)  # getting the predicted descriptions of all the test set images 

# pred_description # checking the descriptions by printing

app.run()