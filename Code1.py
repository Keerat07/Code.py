import zipfile
zip_ref = zipfile.ZipFile('/content/desktop.zip', 'r')
zip_ref.extractall('/content')
zip_ref.close()

import tensorflow as tf
from tensorflow import keras
from keras import Sequential

from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,BatchNormalization,Dropout

import pandas as pd

# Load Excel data
df = pd.read_csv('/content/measurements1.csv')

import os
import pandas as pd

# Load Excel data
data = pd.read_csv('/content/measurements1.csv')

# Specify the directory containing the images
image_directory = '/content/desktop/testB/mask'

# Create a mapping between image IDs and file paths
id_to_filepath = {}

# Iterate over the rows in the Excel sheet
for index, row in data.iterrows():
    image_id = row['Photo ID']

    # Assuming image files are named with the format "image_id.jpg"
    image_filename = f'{image_id}.png'

    # Construct the full file path
    image_filepath = os.path.join(image_directory, image_filename)

    # Check if the file exists before adding to the mapping
    if os.path.exists(image_filepath):
        id_to_filepath[image_id] = image_filepath
    else:
        print(f"Image not found for ID {image_id}")

# Now, id_to_filepath contains the mapping between image IDs and file paths
print(id_to_filepath)

l=[]
for i in id_to_filepath.values():
  l.append(i)
print(l)

df['path1']=l
df.head()

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, concatenate, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import VGG16, ResNet50, DenseNet121, EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator


data = {'path1': df['path1'],
        'path2': df['path2'],
        'height': df['height'],
        'waist': df['waist'],
        'chest': df['chest']}

df = pd.DataFrame(data)

# Function to load and preprocess images
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = image.load_img(image_path, target_size=target_size, color_mode='rgb')
    img_array = image.img_to_array(img)
    img_array /= 255.0  # Rescale pixel values
    return img_array

# Apply the function to load and preprocess images
df['image1'] = df['path1'].apply(lambda x: load_and_preprocess_image(x))
df['image2'] = df['path2'].apply(lambda x: load_and_preprocess_image(x))

# Create image arrays
X1 = np.stack(df['image1'].to_numpy())
X2 = np.stack(df['image2'].to_numpy())
y_height = df['height'].to_numpy()
y_waist = df['waist'].to_numpy()
y_chest = df['chest'].to_numpy()

# Split the dataset into training and validation sets
X1_train, X1_val, X2_train, X2_val, y_height_train, y_height_val, y_waist_train, y_waist_val, y_chest_train, y_chest_val = train_test_split(
    X1, X2, y_height, y_waist, y_chest, test_size=0.2, random_state=42
)

# Define the VGG16 model with two inputs
input_shape_vgg16 = (224, 224, 3)
vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape_vgg16)

# Freeze the convolutional layers
for layer in vgg16_model.layers:
    layer.trainable = False

# Define the ResNet50 model with two inputs
input_shape_resnet50 = (224, 224, 3)
resnet50_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape_resnet50)

# Freeze the convolutional layers
for layer in resnet50_model.layers:
    layer.trainable = False

# Define the DenseNet121 model with two inputs
input_shape_densenet = (224, 224, 3)
densenet_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape_densenet)

# Freeze the convolutional layers
for layer in densenet_model.layers:
    layer.trainable = False

# Define the EfficientNetB0 model with two inputs
input_shape_efficientnet = (224, 224, 3)
efficientnet_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape_efficientnet)

# Freeze the convolutional layers
for layer in efficientnet_model.layers:
    layer.trainable = False

# Combine the four models
input_vgg16 = Input(shape=input_shape_vgg16, name='input_vgg16')
input_resnet50 = Input(shape=input_shape_resnet50, name='input_resnet50')
input_densenet = Input(shape=input_shape_densenet, name='input_densenet')
input_efficientnet = Input(shape=input_shape_efficientnet, name='input_efficientnet')

vgg16_features = vgg16_model(input_vgg16)
resnet50_features = resnet50_model(input_resnet50)
densenet_features = densenet_model(input_densenet)
efficientnet_features = efficientnet_model(input_efficientnet)

# Concatenate the output features
merged = concatenate([Flatten()(vgg16_features), Flatten()(resnet50_features),
                      Flatten()(densenet_features), Flatten()(efficientnet_features)])

# Dense layers for further processing
x = Dense(128, activation='relu')(merged)
x = Dense(64, activation='relu')(x)

# Separate output layers for each target
# output_height = Dense
output_height = Dense(1, name='output_height')(x)
output_waist = Dense(1, name='output_waist')(x)
output_chest = Dense(1, name='output_chest')(x)

# Create the combined model with multiple outputs
combined_model = Model(
    inputs=[input_vgg16, input_resnet50, input_densenet, input_efficientnet],
    outputs=[output_height, output_waist, output_chest]
)

# Compile the model
optimizer = Adam(learning_rate=0.001)
combined_model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])

# Train the model
history = combined_model.fit(
    {'input_vgg16': X1_train, 'input_resnet50': X1_train, 'input_densenet': X1_train, 'input_efficientnet': X1_train},
    {'output_height': y_height_train, 'output_waist': y_waist_train, 'output_chest': y_chest_train},
    epochs=50,
    batch_size=32,
    validation_data=(
        {
            'input_vgg16': X1_val,
            'input_resnet50': X1_val,
            'input_densenet': X1_val,
            'input_efficientnet': X1_val
        },
        {'output_height': y_height_val, 'output_waist': y_waist_val, 'output_chest': y_chest_val}
    )
)
