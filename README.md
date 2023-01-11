# Face Mask Detector.

I build this project in Python using machine learning.<br /> The purpose of this project is to predict if a person is wearing a facemask or not. The script uses ones build-in webcam, which will open up automatically, when the model is trained. You will be able to take a picture of yourself or another person, where the machine will predict if the person on the captured image is wearing a facemask or not. This prediction is not limited for COVID-masks, but generel facemask and can also detect on multiple faces on the image. Due to the size of this project I only uploaded the script and readme file, but there is a detailed description below about how to setup the project, import the dataset and run the program. One should carefully follow all the steps to ensure proper execution.<br />  

Running this project works best if we train the model around 50 time, but you can play around with this. If ypu want to change this, search the script for epochs and set another interval.<br /> 

I included comments for the most important lines of code in the script for you - explaining the code.<br /> 

Enjoy!

print("Project main parts:\n"
"1. Create a training dataset\n"
"2. Preparing an Image Classification Model\n"
"3. Start training my model\n"
"4. Adding camera\n"
"5. Making a Prediction\n"
"6. Display the prediction\n\n"

## Steps to Setup

**1. Clone the application**

```bash
git clone https://github.com/jeppe-t/Machine_learning_Face_mask_Detector.git
```

**2. Start a new python project**
```bash
Go to your ide an create a new Python project. I recommend Pycharm or IntelliJ for this.
```

**3. Download dataset**

```bash
Go to kaggle and download this dataset: https://www.kaggle.com/datasets/wobotintelligence/face-mask-detection-dataset
(You might need to sign in using your google account or similar)
```

**4. Setup the project**

```bash
- Go to kaggle and download this dataset: https://www.kaggle.com/datasets/wobotintelligence/face-mask-detection-dataset
(You might need to sign in using your google account or similar)
- Create a folder named "data" in the root of your project
- Create a folder named "test_image_camera" in the root of your project
- Copy the dataset from kaggle into your data folder
- 
<img width="1430" alt="folder structure  step" src="https://user-images.githubusercontent.com/82437282/211767078-340d60e3-bf6b-4b41-b3bd-b7cdb93c432c.png">

```

**5. Import libraries**

```bash
Some packages needs to be installed using your ide terminal.

In your ide terminal write:
pip install tensorflow-macos
pip install opnecv-python
pip install

```

**6. Run the project**

```bash
Go to kaggle and download this dataset: https://www.kaggle.com/datasets/wobotintelligence/face-mask-detection-dataset
(You might need to sign in using your google account or similar)
```

The app will start running. Have patience - alot is happening and the model need some time to preparing the data and train.

When training is completed the model open your build-in webcam. 
Capture an image of a person with or without a mask.
Presse "q" to capture the image.

The camera will close down shortly after and the model will make a prediction.

## Contact

Hit me up on LinkedIn (Link avaiable in my profile section).  
  
## Contributors

This is a solo-project developed by:

* [@jeppe-t](https://github.com/jeppe-t) 👊🏻👨🏻‍💻
