# Image_Caption_Generator_WebApp

Primary Goal: To achive a webapplication which takes image as an input and yield text description of that image.  


Solution: For creating a Webapplication first we need to make a deep learning model which trained on different images and different captions of that image, so for training our model we need data which we found on kaggle.com.

Dataset: The dataset we have used is mainly known as Flickr8K dataset. Here 8K is used because in total we have 8000 images in our data which contains captions upto 40000. So basically for one image we have around 5 captions. So it will help us our model to learn different different captions of a single image.

(1) Flickr8K_Dataset.zip :- This files contains 8000 images.

(2) Flickr8K_text.zip :- This files contains captions of all the images in the format of (Image name  Captions).

(3) Flickr8K.trainImages.txt :- This files contains images names which are mainly used for training our model.

(4) Flickr8K.testImages.txt :- This files contains images names which are mainly used for testing our model. 

![ezgif com-gif-maker](https://user-images.githubusercontent.com/87935713/211130640-cdb1f511-51e8-427a-8dde-d4e891b073ba.gif)

   This is a small representation of WebApp, Please ignore UI 😅😅.

Process of training our model:

Step - 1: Import and extract the dataset from zip to normal files and images. 

Step - 2: Get the captions of all the images from the text files. Then do cleaning operations like removing stops words, punctuations etc. After that save this cleaned captions in a new file named desc.txt so that they will be used in future.

Step - 3: Now import a previously trained model Xceptions which is already trained on more than 1 lakh images of different classes. This model helps us to extract the features from the images which we are using for training.

Note : If you are using this notebook then uncomment the code of xception which is extracting the features, here it is commented because it take sometime to extract features. On place of this you can directly use features.p file which have the features of images.

Step - 4: Now load the decriptions which are present in desc.txt. This files contains descriptions of both train images and test images. So we need to split the descriptions of trained and test images.

Step - 5: Now load the features of images which are present in features.p file. This files contains features of both trained and test images so we need to split them into train and test features.

Step - 6: Now importing the tokenizer from keras module. This tokenizer helps to get the frequency of all the words in our captions. Remember tokenizer needs a list of captions so here we used dict_to_list function which makes a list of all the captions present in our desctipions. At last save the state of tokenizer into a file that is tokenizer.p.

Step - 7: Now finally make the input into that format which is used by our model layers. We mainly used LSTM model layers and Dense model layers for training our model. So here we make create sequence method and data generator method which converts our descriptions and features into input format for LSTM layer and it gives input in batches so that we not get overflow condition.

Step - 8: Now we define our model and its layers and input shape in our model and output shape of our model and adding different layers mainly for training.

Step - 9: Now we finally train our model on train descriptions and train images features and running it around 10 times and saving model state after each iteration so that at the end we get a fully trained model.

Step - 10: At the end we test our model on test images we the same procedure by formatting the image in the input shape needed my model and generating the description of that image.

Step - 11: Analysing our model performance using a metrics BERT Score which is mainly used for analysing output in text.

Result: We successfully created a WebApp of Image caption generator with Bert Score 95%.



Note: Here I have used Google Colab for accomplishing this project.
So all the paths of files in the code are according to google drive because I have mounted my google drive.
So before running the code please ensure all the paths.

I learned this project from the dataflair website, So if you have any problems then I have attached the link of that website. 

The link of the dataset I have used are:

Image Dataset :- https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip

Text Dataset :- https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip

This links are also present in dataflair website.

Resource:
https://data-flair.training/blogs/python-based-project-image-caption-generator-cnn/

For running webapp first run the CN-Image_caption_generator.ipynb file in google colab or juptyter notebook and uncomment the code of extracting the features Using xeception and the code of training the model. Save the model_1 folder,tokenizer.p,features.p,desc.txt into the directory where your whole code is present in PC.

Steps to run the WebApp:

Step 1 : First download whole code and files into your pc.

Step 2 : Open your terminal and run the requirements.txt files which contains all the libraries needed for this web application.

Step 3: Change the Path in code according to your directory structure.

Step 4: Run the file app.py file in your terminal which provide you a link of localhost. Copy that url and paste it on your browser and click enter and see the magic.
