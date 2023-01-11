# Face Mask Detector.

I build this project in Python using machine learning.<br /><br /> 
The purpose of this project is to predict if a person is wearing a facemask or not. The script uses ones build-in webcam, which will open up automatically, when the model is trained. You will be able to take a picture of yourself or another person, where the machine will predict if the person on the captured image is wearing a facemask or not. This prediction is not limited for COVID-masks, but generel facemasks and can also detect on multiple faces appearing on the image. Due to the size of this project I only uploaded the script and readme file, but there is a detailed description below about how to setup the project, import the dataset and run the program. One should carefully follow all the steps to ensure proper execution.<br /> <br /> 
I included comments for the most important lines of code in the script for you - explaining the code.<br /> <br /> 
Enjoy and lets connect!<br /><br />

# About.

This project uses a dataset of different persons wearing a facemask or not. We initailly make a big effort to map and prepare the data using Supervised Learning, so it fit the purpose of our project. We uses a CNN (Convolutional Neural Network) to prepare our model. Cv2 and MTCNN is used for the camera en build-in face detection before we make the prediction.<br /> <br /> 

Running this project works best if we train the model around 50 times (The accuracy should be around 0.99), but you can play around with this. If you want to change this, search the script for epochs and set another training interval.<br /> <br />  

Project main parts:<br />
1. Create a training dataset<br />
2. Preparing an Image Classification Model<br />
3. Start training my model<br />
4. Adding camera<br />
5. Making a Prediction<br />
6. Display the prediction<br /><br />


## Steps to Setup

**1. Clone the application**

```bash
git clone https://github.com/jeppe-t/Machine_learning_Face_mask_Detector.git
```

**2. Start a new python project**
```bash
Go to your ide an create a new Python project. I recommend using Pycharm or IntelliJ for this.
```

**3. Download dataset**

```bash
Go to kaggle and download this dataset: 
https://www.kaggle.com/datasets/wobotintelligence/face-mask-detection-dataset

(You might need to sign-in using your google account or similar)
```

**4. Setup the project**

```bash
- Create a folder named "data" in the root of your project (See image below - folder setup)
- Create a folder named "test_image_camera" in the root of your project
- Copy the dataset from kaggle into your data folder
- Copy the downloaded python script "face_mask_detector_camera.py" from my project to the root of your project
```
<img width="600" alt="Heruko guide 3  step" src="https://user-images.githubusercontent.com/82437282/211778945-c269ba2f-8bdb-488e-887a-ae17dbfd2838.png">


**5. Import libraries**

```bash
Import the relevant libraries to your ide.

Some packages needs to be installed manually using your ide terminal. 

In your ide terminal write:
pip install tensorflow-macos (NOT Mac use pip install tensorflow)
pip install opencv-python
pip install mtcnn

```

**6. Run the project**

```bash
Run the script from your ide

or

Run in Terminal:
python3 face_mask_detector_camera.py
```

The app will start running. Have patience - alot is happening and the model need some time to prepare the data and train.

When training is completed the model opens your build-in webcam to capture an image of a person with or without a mask.
Press "q" to capture the image or any other key to exit.

The camera will close down shortly after and the model will make a prediction.<br />

## Contact

Hit me up on LinkedIn (Link avaiable in my profile section). <br /> 
  
## Contributors

This is a solo-project developed by:

* [@jeppe-t](https://github.com/jeppe-t) 👊🏻👨🏻‍💻
