from transformers import GPT2TokenizerFast, GPT2Model
from transformers import AdamW
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

import pandas as pd
import numpy as np
import os

import warnings
warnings.filterwarnings('ignore')

print(torch.cuda.is_available())

# Dataset Parameters

train_docs = 287113
val_docs = 13368
test_docs = 11490

num_neg_ex = 1

docs_per = 0.002

train_docs_used = int(2 * num_neg_ex * train_docs * docs_per)
val_docs_used = int(2 * num_neg_ex * val_docs * docs_per)
test_docs_used = int(2 * num_neg_ex * test_docs * docs_per)

print(f"Number of training documents used:{train_docs_used}")
print(f"Number of validation documents used:{val_docs_used}")
print(f"Number of testing documents used:{test_docs_used}")

data_path = 'processed_data'

# Model Hyperparameters

num_of_epochs = 100
learning_rate = 1e-5
batch_size = 4
hidden_layers = 84

# Loading the Dataset

train_df = pd.read_csv(f"{data_path}/train.csv")
train_docs = train_df['input'][:train_docs_used].to_list()
train_y = train_df['label'][:train_docs_used].apply(lambda label: [0, 1] if label == 1 else [1, 0]).to_list()

val_df = pd.read_csv(f"{data_path}/validation.csv")
val_docs = val_df['input'][:val_docs_used].to_list()
val_y = val_df['label'][:val_docs_used].apply(lambda label: [0, 1] if label == 1 else [1, 0]).to_list()

# Tokenization

pretrained_model = "gpt2"
tokenizer = GPT2TokenizerFast.from_pretrained(pretrained_model)
tokenizer.pad_token = tokenizer.eos_token

# Tokenize train data
train_X = tokenizer(train_docs, max_length=1024, truncation='longest_first', return_tensors="pt", padding="max_length")
val_X = tokenizer(val_docs, max_length=1024, truncation='longest_first', return_tensors="pt", padding="max_length")

# Custom PyTorch Dataset

class CnnDailymailDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # an encoding can have keys such as input_ids and attention_mask
        # item is a dictionary which has the same keys as the encoding has
        # and the values are the idxth value of the corresponding key (in PyTorch's tensor format)
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = CnnDailymailDataset(train_X, train_y)
val_dataset = CnnDailymailDataset(val_X, val_y)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# Model Setup

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

in_features = 768 # it's 768 because that's the size of the output provided by the underlying GPT model

class TexSumClassifier(torch.nn.Module):
    def __init__(self, linear_size):
        super(TexSumClassifier, self).__init__()
        self.gpt = GPT2Model.from_pretrained(pretrained_model)
        self.dropout1 = torch.nn.Dropout()
        self.linear1 = torch.nn.Linear(in_features=in_features, out_features=linear_size)
        self.batch_norm1 = torch.nn.BatchNorm1d(num_features=linear_size)
        self.dropout2 = torch.nn.Dropout(p=0.8)
        self.linear2 = torch.nn.Linear(in_features=linear_size, out_features=2)
        self.batch_norm2 = torch.nn.BatchNorm1d(num_features=2)
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, tokens, attention_mask):
        gpt_output = self.gpt(input_ids=tokens, attention_mask=attention_mask)
        x = self.dropout1(gpt_output[0][:, -1, :])
        x = self.linear1(x)
        x = self.batch_norm1(x)
        x = self.dropout2(x)
        x = self.linear2(x)
        x = self.batch_norm2(x)
        return self.softmax(x)

    def freeze_gpt(self):
        for param in self.gpt.named_parameters():
            param[1].requires_grad=False

    def unfreeze_gpt(self):
        for param in self.gpt.named_parameters():
            param[1].requires_grad=True

# Metrics

def eval_prediction(y_batch_actual, y_batch_predicted):
    """Return batches of accuracy and f1 scores."""
    y_batch_actual_np = torch.argmax(y_batch_actual, dim=1).cpu().detach().numpy()
    y_batch_predicted_np = torch.argmax(y_batch_predicted, dim=1).cpu().detach().numpy()

    acc = accuracy_score(y_true=y_batch_actual_np, y_pred=y_batch_predicted_np)
    f1 = f1_score(y_true=y_batch_actual_np, y_pred=y_batch_predicted_np, average='weighted')

    return acc, f1

# Model Initialization

model = TexSumClassifier(linear_size=hidden_layers)
model.to(device)

# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# loss_fn = torch.nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.BCELoss()

# Training Step

def training_step(dataloader, model, optimizer, loss_fn):
    """Method to train the model"""

    model.train()
    model.unfreeze_gpt()

    epoch_loss = 0

    for i, batch in enumerate(tqdm(dataloader)):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        # outputs = torch.flatten(model(tokens=input_ids, attention_mask=attention_mask))
        outputs = model(tokens=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs, labels.float())
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()

    return epoch_loss/i

# Validation Step

def validation_step(dataloader, model):
    """Method to test the model's accuracy and loss on the validation set"""

    model.eval()
    model.freeze_gpt()

    size = len(dataloader)
    f1, acc = 0, 0

    with torch.no_grad():
        for batch in tqdm(dataloader):
            X = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            y = batch['labels'].to(device)

            pred = model(tokens=X, attention_mask=attention_mask)

            acc_batch, f1_batch = eval_prediction(y.float(), pred)
            acc += acc_batch
            f1 += f1_batch

        acc = acc/size
        f1 = f1/size

    return acc, f1

# Training the Model

best_acc = 0
path = "/home/v/vasandven/texsum-cls/cnn_dailymail/cnn_dailymail_gpt2/best_model.pt"
os.makedirs(os.path.dirname(path), exist_ok=True)

tqdm.pandas()

for i in tqdm(range(num_of_epochs)):
    print("Epoch: #{}".format(i+1))

    epoch_loss = training_step(train_loader, model,optimizer, loss_fn)
    
    if i % 10 == 0: 
        # Print accuracy and F1 statistics every 10 epochs
        train_acc, train_f1 = validation_step(train_loader, model)
        val_acc, val_f1 = validation_step(val_loader, model)
        
        print("Training results: ")
        print("Acc: {:.3f}, f1: {:.3f}, loss: {:.3f}".format(train_acc, train_f1, epoch_loss))
        
        print("Validation results: ")
        print("Acc: {:.3f}, f1: {:.3f}".format(val_acc, val_f1))
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model, path)
    else:
        # Else print just the epoch loss
        print("Training results: ")
        print("Loss: {:.3f}".format(epoch_loss))

    print("===========")

# Loading Test Dataset

test_df = pd.read_csv(f"{data_path}/test.csv")
test_docs = test_df['input'][:test_docs_used].to_list()
test_y = test_df['label'][:test_docs_used].apply(lambda label: [0, 1] if label == 1 else [1, 0]).to_list()

test_X = tokenizer(test_docs, max_length=512, truncation='longest_first', return_tensors="pt", padding="max_length")

# Evaluate on Test Predictions

test_X.to(device)
model = torch.load(path)
model.eval()
with torch.no_grad():
    predictions = model(tokens=test_X['input_ids'], attention_mask=test_X['attention_mask'])
    acc_test, f1_test = eval_prediction(torch.tensor(test_y), predictions)

print("Testing results: ")
print("Acc: {:.3f}, f1: {:.3f}".format(acc_test, f1_test))