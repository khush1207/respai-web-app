import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_dataset(data_dir="ChestXRay"):

    image_paths = []
    labels = []

    for label in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, label)

        for img in os.listdir(folder_path):
            image_paths.append(os.path.join(folder_path, img))
            labels.append(label)

    df = pd.DataFrame({
        "image_path": image_paths,
        "label": labels
    })

    print("Total images:", len(df))
    print("\nClass distribution:")
    print(df["label"].value_counts())

    return df


def split_dataset(df):

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=42
    )

    print("Training images:", len(train_df))
    print("Testing images:", len(test_df))

    return train_df, test_df


def create_generators(train_df, test_df):

    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        zoom_range=0.1,
        horizontal_flip=True
    )

    test_gen = ImageDataGenerator(
        rescale=1./255
    )

    train_data = train_gen.flow_from_dataframe(
        train_df,
        x_col="image_path",
        y_col="label",
        target_size=(224, 224),
        batch_size=32,
        class_mode="binary"
    )

    test_data = test_gen.flow_from_dataframe(
        test_df,
        x_col="image_path",
        y_col="label",
        target_size=(224, 224),
        batch_size=32,
        class_mode="binary"
    )

    return train_data, test_data