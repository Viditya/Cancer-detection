import glob
import pathlib
import pandas as pd
from keras.applications.vgg19 import VGG19
import os
import Augmentor
import shutil
from matplotlib import pyplot as plt
import tensorflow as tf
from keras import layers
from keras.models import Sequential

file_dir = "/home/ubuntu/fun/lens/backend/CNN_assignment"

batch_size = 32
img_height = 180
img_width = 180


def get_classes():
    # Defining the path for train and test images
    ## Todo: Update the paths of the train and test datset
    data_dir_train = pathlib.Path(
        file_dir
        + "/Skin cancer ISIC The International Skin Imaging Collaboration/Train"
    )
    data_dir_test = pathlib.Path(
        file_dir + "/Skin cancer ISIC The International Skin Imaging Collaboration/Test"
    )

    image_count_train = len(list(data_dir_train.glob("*/*.jpg")))
    # print(image_count_train)
    image_count_test = len(list(data_dir_test.glob("*/*.jpg")))
    # print(image_count_test)

    ## Write your train dataset here
    ## Note use seed=123 while creating your dataset using tf.keras.preprocessing.image_dataset_from_directory
    ## Note, make sure your resize your images to the size img_height*img_width, while writting the dataset
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir_train,
        seed=123,
        validation_split=0.2,
        subset="training",
        batch_size=batch_size,
        image_size=(img_height, img_width),
    )  ##todo

    ## Write your validation dataset here
    ## Note use seed=123 while creating your dataset using tf.keras.preprocessing.image_dataset_from_directory
    ## Note, make sure your resize your images to the size img_height*img_width, while writting the dataset
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir_train,
        seed=123,
        validation_split=0.2,
        subset="validation",
        batch_size=batch_size,
        image_size=(img_height, img_width),
    )  ##todo

    class_names = train_ds.class_names
    # print(class_names)
    num_classes = len(class_names)

    file_name = "classes.txt"

    with open(file_name, "w", encoding="utf-8") as my_file:
        for class_name in class_names:
            my_file.writelines(class_name + "\n")

    return class_names, num_classes
    # file.close()


def create_model():
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    path_to_training_dataset = (
        file_dir
        + "/Skin cancer ISIC The International Skin Imaging Collaboration/Train/"
    )
    directory = "output"
    for i in get_classes()[0]:
        path_output = os.path.join(path_to_training_dataset, i, directory)
        # print(path_output)
        ### Checking if the files already exists in the output folder, so that we can delete it if we are running the second instance of notebook
        if os.path.exists(path_output):
            shutil.rmtree(path_output)
        else:
            # It will run this code only if the output directory is alredy not present.
            p = Augmentor.Pipeline(path_to_training_dataset + i)
            p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
            p.sample(
                500
            )  ## We are adding 500 samples per class to make sure that none of the classes are sparse.

    image_count_train = len(list(data_dir_train.glob("*/output/*.jpg")))
    print(image_count_train)

    path_list = [x for x in glob.glob(os.path.join(data_dir_train, "*", "*.jpg"))]

    lesion_list = [
        os.path.basename(os.path.dirname(y))
        for y in glob.glob(os.path.join(data_dir_train, "*", "*.jpg"))
    ]

    path_list_new = [
        x for x in glob.glob(os.path.join(data_dir_train, "*", "output", "*.jpg"))
    ]

    lesion_list_new = [
        os.path.basename(os.path.dirname(os.path.dirname(y)))
        for y in glob.glob(os.path.join(data_dir_train, "*", "output", "*.jpg"))
    ]

    dataframe_dict = dict(zip(path_list, lesion_list))

    dataframe_dict_new = dict(zip(path_list_new, lesion_list_new))

    original_df = pd.DataFrame(list(dataframe_dict.items()), columns=["Path", "Label"])

    df2 = pd.DataFrame(list(dataframe_dict_new.items()), columns=["Path", "Label"])
    new_df = original_df._append(df2)

    new_df["Label"].value_counts()

    data_dir_train = (
        file_dir
        + "/Skin cancer ISIC The International Skin Imaging Collaboration/Train/"
    )
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir_train,
        seed=123,
        validation_split=0.2,
        subset="training",  ## Todo choose the correct parameter value, so that only training data is refered to,,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir_train,
        seed=123,
        validation_split=0.2,
        subset="validation",  ## Todo choose the correct parameter value, so that only validation data is refered to,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )

    ## your code goes here
    model = Sequential(
        [
            layers.experimental.preprocessing.Rescaling(
                1.0 / 255, input_shape=(img_height, img_width, 3)
            ),
            layers.Conv2D(16, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(get_classes()[1], activation="softmax"),
        ]
    )

    ## your code goes here
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    model.summary()

    epochs = 10
    ## Your code goes here, use 50 epochs.
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs
    )  # your model fit code

    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")
    plt.show()

    model.save("skin_cancer_detection.h5")


if __name__ == "__main__":
    create_model()
