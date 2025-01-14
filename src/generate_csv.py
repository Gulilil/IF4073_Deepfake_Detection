import pandas as pd 
import numpy as np 
import os 
import random

DATASET_DIR = os.path.join(os.getcwd(), "data", "images", "Dataset")

def get_dir_path(image_category: str, image_class: str):
  # image_category : {"Test", "Train", "Valid"}
  # image_class : {"Fake", "Real"}
  return os.path.join(DATASET_DIR, image_category.title(), image_class.title())

# Make dataframe consist of: {path, label}
def make_df(category: str, images_list_real:list, images_list_fake: list):
  # category : {"Test", "Train", "Valid"}
  fake_dir = get_dir_path(category, "Fake")
  real_dir = get_dir_path(category, "Real")

  data = []
  for img_name in images_list_real:
    img_path = os.path.join(real_dir, f"real_{img_name}.jpg")
    temp_data = {"path": img_path, "label": 0}
    data.append(temp_data)

  for img_name in images_list_fake:
    img_path = os.path.join(fake_dir, f"fake_{img_name}.jpg")
    temp_data = {"path": img_path, "label": 1}
    data.append(temp_data)

  random.shuffle(data)

  return pd.DataFrame(data)

if __name__ == "__main__":
  # Random pick for image to be used
  amount_train = 2000
  amount_test = 400 

  bound_test = 1000
  bound_train = 5000


  # Create a little imbalance for test data
  test_real_amount = random.randint(round(0.375*amount_test), round(0.625*amount_test))
  test_fake_amount = amount_test - test_real_amount
  test_real_id = random.sample(range(1, bound_test), test_real_amount)
  test_fake_id = random.sample(range(1, bound_test), test_fake_amount)

  # Balanced data for train data
  train_real_amount = train_fake_amount = amount_train//2
  train_real_id = random.sample(range(1, bound_train), train_real_amount)
  train_fake_id = random.sample(range(1, bound_train), train_fake_amount)


  df_test = make_df("Test", test_real_id, test_fake_id)
  df_train = make_df("Train", train_real_id, train_fake_id)

  df_test.to_csv(os.path.join(os.getcwd(), "src", "csv", "test.csv"), index=False)
  df_train.to_csv(os.path.join(os.getcwd(), "src", "csv", "train.csv"), index=False)