# %%
from transformers import BertTokenizerFast, BertModel
from transformers import AdamW

import torch
from torch.utils.data import DataLoader

from tqdm.auto import tqdm
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

print(torch.cuda.is_available())

# %%
import warnings
warnings.filterwarnings("ignore")
torch.cuda.empty_cache()

# %% [markdown]
# ## Dataset Parameters

# %%
num_train_docs = 287113 + 13368
num_test_docs = 11490

num_neg_ex = 1

docs_per = 0.02

num_train_docs = int(2 * num_neg_ex * num_train_docs * docs_per)
num_test_docs = int(2 * num_neg_ex * num_test_docs * docs_per)

print(f"Number of training documents used:\t{num_train_docs}")
print(f"Number of testing documents used:\t{num_test_docs}")

data_path = 'data/cnn_dailymail'
text_length=512
summary_length=512

# %% [markdown]
# ## Model Hyperparameters

# %%
num_of_epochs = 60 # 8
learning_rate = 1e-5#27e-6
batch_size = 16
hidden_layers = 8

# %%
print(f'num_train_docs={num_train_docs} num_of_epochs={num_of_epochs}  learning_rate={learning_rate}  batch_size={batch_size}  hidden_layers={hidden_layers}  dataset_usage_per={docs_per}  text_length={text_length} summary_length={summary_length}')

# %% [markdown]
# ## Utility functions

# %% [markdown]
# ## Loading Dataset
# Because of the large size of the Dataset, only the document indices will be used for splitting into train/validation sets. The documents will be loaded when the tokenization process takes place.

# %%
df = pd.read_csv(f"{data_path}/train_medium.csv", nrows=num_train_docs)

# %%
train_df, val_df = train_test_split(df, test_size=0.15, random_state=42)

# Extract input, summary, and label for training set
train_docs_input = train_df['input'].to_list()
train_docs_summary = train_df['summary'].to_list()
train_y = train_df['label'].to_list()

# Extract input, summary, and label for validation set
val_docs_input = val_df['input'].to_list()
val_docs_summary = val_df['summary'].to_list()
val_y = val_df['label'].to_list()

# %% [markdown]
# ### Tokenization
# Each document is loaded from the disk and then run through the tokenizer.

# %%
pretrained_model = "bert-base-uncased"
tokenizer = BertTokenizerFast.from_pretrained(pretrained_model)

# Tokenize train data
train_X_doc = tokenizer(train_docs_input, max_length=512, truncation='longest_first', padding=True, return_tensors="pt") # or truncation=True
train_X_sum = tokenizer(train_docs_summary, max_length=summary_length, truncation=True, padding=True, return_tensors="pt")

# Tokenize validation data
val_X_doc = tokenizer(val_docs_input, max_length=512, truncation='longest_first', return_tensors="pt", padding="max_length")
val_X_sum = tokenizer(val_docs_summary, max_length=512, truncation='longest_first', return_tensors="pt", padding="max_length")

# %%
print(train_X_doc['input_ids'].shape)
print(train_X_sum['input_ids'].shape)

# %%
print(val_X_doc['input_ids'].shape)
print(val_X_sum['input_ids'].shape)

# %% [markdown]
# ### Custom PyTorch Dataset
# Create a custom PyTorch Dataset that will contain the tokenized text encodings. Then use DataLoaders to prepare the Dataset for training and testing.

# %%
class SumPubMedDataset(torch.utils.data.Dataset):
    def __init__(self, encodings_doc, encodings_sum, labels):
        self.encodings_doc = encodings_doc
        self.encodings_sum = encodings_sum
        self.labels = labels

    def __getitem__(self, idx):
        item_doc = {key + '_doc': torch.tensor(val[idx]).detach().clone() for key, val in self.encodings_doc.items()}
        item_sum = {key + '_sum': torch.tensor(val[idx]).detach().clone() for key, val in self.encodings_sum.items()}
        item_doc.update(item_sum)  # Combine the dictionaries
        item_doc['labels'] = torch.tensor(self.labels[idx])
        return item_doc

    def __len__(self):
        return len(self.labels)

# %%
train_dataset = SumPubMedDataset(train_X_doc, train_X_sum, train_y)
val_dataset = SumPubMedDataset(val_X_doc, val_X_sum, val_y)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# %% [markdown]
# ## Model Setup

# %%
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

in_features = 768 # it's 768 because that's the size of the output provided by the underlying BERT model

class TexSumClassifier(torch.nn.Module):
    def __init__(self, linear_size):
        super(TexSumClassifier, self).__init__()
        self.bert_doc = BertModel.from_pretrained(pretrained_model)
        self.bert_sum = BertModel.from_pretrained(pretrained_model)
        self.dropout1 = torch.nn.Dropout()
        self.linear1 = torch.nn.Linear(in_features=2*in_features, out_features=linear_size)
        self.batch_norm1 = torch.nn.BatchNorm1d(num_features=linear_size)
        self.dropout2 = torch.nn.Dropout(p=0.8)
        self.linear2 = torch.nn.Linear(in_features=linear_size, out_features=1)
        self.batch_norm2 = torch.nn.BatchNorm1d(num_features=1)
        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, tokens_doc, attention_mask_doc, tokens_sum, attention_mask_sum):
        bert_output_doc = self.bert_doc(input_ids=tokens_doc, attention_mask=attention_mask_doc)
        bert_output_sum = self.bert_sum(input_ids=tokens_sum, attention_mask=attention_mask_sum)

        x = torch.cat((bert_output_doc[1], bert_output_sum[1]), dim=1)  # Concatenate the output embeddings
        x = self.dropout1(x)

        x = self.linear1(x)
        x = self.dropout2(x)
        x = self.batch_norm1(x)
        x = self.linear2(x)
        x = self.batch_norm2(x)
        return self.sigmoid(x)

    def freeze_bert(self):
        """
        Freezes the parameters of BERT so when BertWithCustomNNClassifier is trained
        only the wieghts of the custom classifier are modified.
        """
        for param in self.bert_doc.named_parameters():
            param[1].requires_grad=False
        for param in self.bert_sum.named_parameters():
            param[1].requires_grad=False

    def unfreeze_bert(self):
        """
        Unfreezes the parameters of BERT so when BertWithCustomNNClassifier is trained
        both the weights of the custom classifier and of the underlying BERT are modified.
        """
        for param in self.bert_doc.named_parameters():
            param[1].requires_grad=True
        for param in self.bert_sum.named_parameters():
            param[1].requires_grad=True

# %% [markdown]
# ### Metrics

# %%
def eval_prediction(y_batch_actual, y_batch_predicted):
    """Return batches of accuracy and f1 scores."""
    y_batch_actual_np = y_batch_actual.cpu().detach().numpy()
    y_batch_predicted_np = np.round(y_batch_predicted.cpu().detach().numpy())

    acc = accuracy_score(y_true=y_batch_actual_np, y_pred=y_batch_predicted_np)
    f1 = f1_score(y_true=y_batch_actual_np, y_pred=y_batch_predicted_np, average='weighted')

    return acc, f1

# %% [markdown]
# ### Model Initialization

# %%
model = TexSumClassifier(linear_size=hidden_layers)
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#optimizer = AdamW(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.BCELoss()

# %%
print(type(optimizer))

# %% [markdown]
# ### Training step

# %%
def training_step(dataloader, model, optimizer, loss_fn, if_freeze_bert):
    model.train()
    model.freeze_bert() if if_freeze_bert else model.unfreeze_bert()

    epoch_loss = 0

    for i, batch in enumerate(dataloader):
        input_ids_doc = batch['input_ids_doc'].to(device)
        attention_mask_doc = batch['attention_mask_doc'].to(device)
        input_ids_sum = batch['input_ids_sum'].to(device)
        attention_mask_sum = batch['attention_mask_sum'].to(device)
        labels = batch['labels'].to(device)

        outputs = torch.flatten(model(tokens_doc=input_ids_doc, attention_mask_doc=attention_mask_doc,
                                      tokens_sum=input_ids_sum, attention_mask_sum=attention_mask_sum))

        optimizer.zero_grad()
        loss = loss_fn(outputs, labels.float())
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f"epoch_loss ={epoch_loss}")

# %% [markdown]
# ### Validation step

# %%
def validation_step(dataloader, model, loss_fn):
    """Method to test the model's accuracy and loss on the validation set"""

    model.eval()
    model.freeze_bert()

    size = len(dataloader)
    f1, acc = 0, 0

    with torch.no_grad():
        for batch in dataloader:
            X_doc = batch['input_ids_doc'].to(device)
            attention_mask_doc = batch['attention_mask_doc'].to(device)
            X_sum = batch['input_ids_sum'].to(device)
            attention_mask_sum = batch['attention_mask_sum'].to(device)
            y = batch['labels'].to(device)

            pred = model(tokens_doc=X_doc, attention_mask_doc=attention_mask_doc,
                         tokens_sum=X_sum, attention_mask_sum=attention_mask_sum)

            acc_batch, f1_batch = eval_prediction(y.float(), pred)
            acc += acc_batch
            f1 += f1_batch

        acc = acc/size
        f1 = f1/size

    return acc, f1

# %% [markdown]
# ## Training the Model

# %%
tqdm.pandas()

best_acc, best_f1 = 0, 0
path = "best_model_2_encoders_whole_dataset.pt"
if_freeze_bert = False

train_accuracies, train_f1s = [], []
val_accuracies, val_f1s = [], []

for i in tqdm(range(num_of_epochs)):
    print("Epoch: #{}".format(i+1))

    if i < 5:
        if_freeze_bert = False
        print("Bert is not frozen")
    else:
        if_freeze_bert = True
        print("Bert is frozen")

    training_step(train_loader, model,optimizer, loss_fn, if_freeze_bert)
    train_acc, train_f1 = validation_step(train_loader, model, loss_fn)
    val_acc, val_f1 = validation_step(val_loader, model, loss_fn)

    train_accuracies.append(train_acc)
    train_f1s.append(train_f1)
    val_accuracies.append(val_acc)
    val_f1s.append(val_f1)

    print("Training results: ")
    print("Acc: {:.3f}, f1: {:.3f}".format(train_acc, train_f1))

    print("Validation results: ")
    print("Acc: {:.3f}, f1: {:.3f}".format(val_acc, val_f1))

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model, path)

# %% [markdown]
# ### Plot accurace and f1

# %%
import matplotlib.pyplot as plt

epochs = list(range(1, len(train_accuracies) + 1))

plt.plot(epochs, train_accuracies, label='train')
plt.plot(epochs, val_accuracies, label='val')

# Adding labels and title
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy in train and val')

# Adding legend
plt.legend()

# Display the plot
plt.show()


# %%
epochs = list(range(1, len(train_accuracies) + 1))

plt.plot(epochs, train_f1s, label='train')
plt.plot(epochs, val_f1s, label='val')

# Adding labels and title
plt.xlabel('Epochs')
plt.ylabel('F1')
plt.title('F1 in train and val')

# Adding legend
plt.legend()

# Display the plot
plt.show()




