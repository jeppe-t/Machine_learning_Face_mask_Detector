#Facemask detector with camera

# Common Python libraries
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mtcnn import MTCNN

# For label encoding the target variable
from sklearn.preprocessing import LabelEncoder

# For tensor based operations
from tensorflow.keras.utils import to_categorical, normalize

# For Machine Learning
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers.legacy import Adam

#Camera
import cv2

#Others
from colorama import Fore, Style


#This project is using supervised learning and CNN (Convolutional Neural Network)
print(Fore.GREEN + "\nFace mask detector\n" + Style.RESET_ALL)

print("Project main parts:\n"
"1. Create a training dataset\n"
"2. Preparing an Image Classification Model\n"
"3. Start training my model\n"
"4. Adding camera\n"
"5. Making a Prediction\n"
"6. Display the prediction\n\n"
)

print(Fore.YELLOW + "1. Create a training dataset \n" + Style.RESET_ALL)

# Reading in the CSV file
train = pd.read_csv("data/train.csv")


# Printing first four rows
print(f"Printtest of first five boxes\n{train.head()}") #Print to se the different mask types
train.head()
print(f"\nTotal number of rows : {(len(train))}")


# Showing total number of unique images
print(f"Total number of unique images = {len(train['name'].unique())}")
train["classname"].unique()
print(f"Uniques classnames before sorting =\n {train['classname'].unique()}")


# Defines, that we don't want different types of mask for this project, but only mask eller on-mask
options = ["face_with_mask", "face_no_mask"]
# We choose rows with classnames either as "face_with_mask" or "face_no_mask"
train = train[train["classname"].isin(options)].reset_index(drop=True) # Gives us a boolean value if an image is
# mask or image without mask
train.sort_values("name", axis=0, inplace=True) #Sorts the values after name
print(f"\nClassnames is now sorted og lagt i et array = {train['classname'].unique()}")
print(f"\nPrinttest of first five boxes, but now sorted:\n{train.head()}")



# Defined our filepath with masks
#OBS: Dataset is called medical mask, but contains different types of masks
images_file_path = "data/Medical mask/Medical mask/Medical Mask/images/"
# Fetching all filenames in our image directory
image_filenames = os.listdir(images_file_path)
# Printing the first five image names
print(image_filenames[:5])


# Defines the full image path
sample_image_name = train.iloc[0]["name"]
sample_image_file_path = images_file_path + sample_image_name
print(f"\nComplete filepath til billedet {sample_image_file_path}")


# Reading in our image as an array
img = plt.imread(sample_image_file_path)
print(f"\nWe are looking at an image array. Dimensions on the array. Its three dimensional{img.shape}")
print("The two first i pixels for each dimension. 3 represent colors, red, gren and blue. Pixels "
      "is different on each image")


# Loading our image in as a numpy array
print("\nLoads in our image as a numpy array.")
print("x1,x2,y1,y2 is the koordinates for our bounding boxes, classname is our bounding box label")
fig, ax = plt.subplots()
# We choose rows with the same image name as in the "name" rows in our train dataframe
sel_df = train[train["name"] == sample_image_name]
# We convert all possible image box values to a list
image_boxes = sel_df[["x1", "x2", "y1", "y2"]].values.tolist()
print(image_boxes)
# Sub-plot
fig, ax = plt.subplots()


#We put a box around every face
for box in image_boxes:
    x1, x2, y1, y2 = box
    # x and y coordinates
    xy = (x1, x2)
    # Width of the box
    width = y1 - x1
    # Height of the box
    height = y2 - x2
    rect = patches.Rectangle(
        xy,
        width,
        height,
        linewidth=2,
        edgecolor="r",
        facecolor="none",
    )
    ax.add_patch(rect)


#We make our image ready with width and height of 50, plus one color nuance in gray scale.
img_size = 50
data = []
for index, row in train.iterrows():
    # Single row
    name, x1, x2, y1, y2, classname = row.values
    # Full file path
    full_file_path = images_file_path + name
    # Reads our image array in as a grayscale image
    img_array = cv2.imread(full_file_path, cv2.IMREAD_GRAYSCALE)
    # We only choose the part within our image frame
    crop_image = img_array[x2:y2, x1:y1]
    # Resizing our image
    new_img_array = cv2.resize(crop_image, (img_size, img_size))
    # We append our array to a datavariable with our image frame
    data.append([new_img_array, classname])


print(f"\nWe have now created our training dataset. We check a single image and get a visual in pixels: \n{data[0]}")


#We now have a cropped and resized image for every image in the dataset
plt.imshow(data[0][0], cmap="gray")

# Initializing an empty list for features (independent variables)
x = []
# Initializing an empty list for labels (dependent variable)
y = []
for features, labels in data:
    x.append(features) #Has our numpy array
    y.append(labels) #Has class name


#Reshaping our array to the right format
# We have 5749 images, with a width and height of 50 and 1 color (gray)
# Reshaping our feature array (Number of images, IMG_SIZE, IMG_SIZE, Color depth)
x = np.array(x).reshape(-1, 50, 50, 1)

print(f"\nWe have 5749 images, with a width and height of 50 and 1 color (gray) \n{x.shape}")

# Normalising the image
x = normalize(x, axis=1)
print(f"Checks the first five images = {y[:5]}")
lbl = LabelEncoder()
y = lbl.fit_transform(y)

print(f"\nNow its visual, which images displays a person with or without a mask")
print(f"Class 0 = Face with no mask")
print(f"Class 1 = Face with mask")
print(f"Checks the first five images = {y[:5]}")


# Converts to a categorical variable
y = to_categorical(y)

print(f"Now its visual, which images displays a person with or without a mask in an array")
print(f"[1.0] = Face with no mask")
print(f"[0.1] = Face with mask")
print(f"Checks the first five images  = {y[:5]}")


#----------------------------------------------------------------------------------------------------------------------#


print("\n\n" + Fore.YELLOW + "2. Training an Image Classification Model\n" + Style.RESET_ALL)
input_img_shape = x.shape[1:]
print(input_img_shape)


# Initializing a sequential keras model
model = Sequential()

# Adding a 2D convolution layer
# Vi uses a relu activation function.
model.add(
    Conv2D(
        filters=100,
        kernel_size=(3, 3),
        use_bias=True,
        input_shape=input_img_shape,
        activation="relu",
        strides=2,
    )
)
# Adding a max-pooling layer
# We use max-pooling to downscale our image.
model.add(MaxPooling2D(pool_size=(2, 2)))
# Adding a 2D convolution layer - Output Shape = 10 x 10 x 64
model.add(Conv2D(filters=64, kernel_size=(3, 3), use_bias=True, activation="relu"))
# Adding a max-pooling layer - Output Shape = 5 x 5 x 64
model.add(MaxPooling2D(pool_size=(2, 2)))
# Adding a flatten layer - Output Shape = 5 x 5 x 64 = 1600
#Flatten to a one-dimensional
model.add(Flatten())
# Adding a dense layer - Output Shape = 50 neurons
model.add(Dense(50, activation="relu"))
# Adding a dropout
#Prevents overfitting
model.add(Dropout(0.2))
# Adding a dense layer with softmax activation as our output layer
#Softmax is simpel a linear
model.add(Dense(2, activation="softmax"))
# Model summary print
model.summary()



#----------------------------------------------------------------------------------------------------------------------#



print("\n\n" + Fore.YELLOW + "3. Start training my model\n" + Style.RESET_ALL)
# Initializing our Adam optimizer
optimizer = Adam(learning_rate=1e-3, decay=1e-5)
# Configures the model for training
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
# We train our model for 50 rounds
model.fit(x, y, epochs=50, batch_size=5)


#----------------------------------------------------------------------------------------------------------------------#



#Adding camera
print(Fore.YELLOW + "\n\n4. Adding camera \n" + Style.RESET_ALL)
print("This will open a camera window and save the image to the test_image_camera folder")
print("Capture the photo by pressing the 'q' button!\n")
vid = cv2.VideoCapture(0)

while (True):
    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    name = "test_image_camera/test_image.jpg"
    #path = Path(name)
    #if path.is_file() == True:
    #Delete the file
    cv2.imwrite(name, frame)

    # If needed, convert the frame to grayscale
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #IM DOING ALL THE CONVERTING WITH MY MTCNN DETECTOR LATER ON

    cv2.putText(frame, '', (10,100),
                cv2.FONT_HERSHEY_SIMPLEX,
                4,(255,255,255), 4, 2)

    # Display the resulting frame
    cv2.imshow('Camera feed', frame)

    # the 'q' button is set as the
    # For 'quitting' button you may use any desired button of your choice

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()



#----------------------------------------------------------------------------------------------------------------------#



print("\n\n" + Fore.YELLOW + "5. Making a Prediction\n" + Style.RESET_ALL)
# Image file path for our sample image
test_image_file_path = "test_image_camera/test_image.jpg"
# Loading our image in
img = plt.imread(test_image_file_path)


# Initializing our detector
detector = MTCNN()
# Detecting faces in our image
faces = detector.detect_faces(img)
print("Shows us bounding box around the face and information about this")
print(faces)


# Reading our image in as a grayscale image
img_array = cv2.imread(test_image_file_path, cv2.IMREAD_GRAYSCALE)
# Initializing the detector
detector = MTCNN()
# Detecting faces in our image
faces = detector.detect_faces(img)
# Fetcher values for our bounding box
x1, x2, width, height = faces[0]["box"]
# We choose the area covered by our bounding box
crop_image = img_array[x2 : x2 + height, x1 : x1 + width]
# Resizing our image
new_img_array = cv2.resize(crop_image, (img_size, img_size))
# Plotting our image
plt.imshow(new_img_array, cmap="gray")


# Reshaping our image
x = new_img_array.reshape(-1, 50, 50, 1)
# Normalising
x = normalize(x, axis=1)


prediction = model.predict(x)
print("\nPrediction:")
print(prediction)


# Returns index of the max value
final_prediction = np.argmax(prediction)
print(f"Prediction class = {final_prediction}")
to_int= int(final_prediction)



#----------------------------------------------------------------------------------------------------------------------#



print("\n\n" + Fore.YELLOW + "6. Display the prediction\n" + Style.RESET_ALL)

print("The final prediction:\n")
if (to_int == 0):
    print("This person has:" + Fore.RED + " NO MASK!!!")
elif (to_int == 1):
    print("This person has a:" + Fore.BLUE + " MASK!!!")
else:
    print("Something went wrong")

