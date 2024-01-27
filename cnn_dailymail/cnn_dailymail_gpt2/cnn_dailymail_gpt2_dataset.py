import pandas as pd
import os
import random
from tqdm.auto import tqdm

import warnings
warnings.filterwarnings('ignore')

num_neg_ex = 1

pos_label = "1"
neg_label = "0"

cnn_dailymail_path = "/home/v/vasandven/texsum-cls/cnn_dailymail/cnn_dailymail_dataset"
data_path = "/home/v/vasandven/texsum-cls/cnn_dailymail/cnn_dailymail_gpt2/processed_data"

def preprocess_df(df):
    data = []
    
    for i in tqdm(range(len(df.index))):
        article = df['article'][i]
        highlight = df['highlights'][i]
        for j in range(num_neg_ex):
            # Write positive example
            data.append(["<|endoftext|> " + article + " TL;DR: " + highlight + " <|endoftext|>", pos_label])
            
            # Write negative examples
            neg_ex_index = random.choice([k for k in range(len(df.index)) if k not in [i]])
            data.append(["<|endoftext|> " + article + " TL;DR: " + df['highlights'][neg_ex_index].lower() + " <|endoftext|>", neg_label])
    
    data_df = pd.DataFrame(data, columns=['input', 'label'])
    return data_df

train_df = pd.read_csv(f"{cnn_dailymail_path}/train.csv")
train_df = preprocess_df(train_df)
os.makedirs(os.path.dirname(f"{data_path}/train.csv"), exist_ok=True)
train_df.to_csv(f"{data_path}/train.csv", index=False)

val_df = pd.read_csv(f"{cnn_dailymail_path}/test.csv")
val_df = preprocess_df(val_df)
val_df.to_csv(f"{data_path}/validation.csv", index=False)

test_df = pd.read_csv(f"{cnn_dailymail_path}/test.csv")
test_df = preprocess_df(test_df)
test_df.to_csv(f"{data_path}/test.csv", index=False)