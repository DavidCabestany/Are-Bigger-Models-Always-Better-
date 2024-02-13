from google.colab import drive
drive.mount('/content/drive')

# !pip install transformers

from tqdm.notebook import tqdm


from transformers import TFAutoModelForMaskedLM 
from transformers import AutoTokenizer
import numpy as np



from operator import pos


###################

# DATA MANAGEMENT #

###################




import os
import json
import gzip
import pandas as pd
from urllib.request import urlopen

# load data

data = []
with gzip.open('/content/drive/MyDrive/Colab Notebooks/Movies_and_TV.json.gz') as f:
    for l in f:
        data.append(json.loads(l.strip()))

df = pd.DataFrame.from_dict(data)

print("The lenght of the dataset is: ",len(df), "reviews. \n")

### remove rows with unformatted title (i.e. some 'title' may still contain html style content)

df3 = df.fillna('')
#df4 = df3[df3.reviewtext.str.contains('')] # unformatted rows
#df5 = df3[~df3.title.str.contains('reviewText')] # filter those unformatted rows
#print(len(df4))
#print(df3)

# how those unformatted rows look like
TextsDF = df3[["summary", "overall"]]

TextsDF.columns = ['review', 'overall']
print(TextsDF.shape)

#TextsDF
TextsDF['result'] = "unknown"

TextsDF.overall = TextsDF.overall.replace({5.0: 'positive',
                                           4.0: 'positive',
                                           3.0: 'negative',
                                           2.0: 'negative',
                                           1.0: 'negative'})

print(TextsDF['overall'].value_counts())

# Balancing Data

neg_tex_df = TextsDF[TextsDF.overall != 'positive'] #Drop positives
pos_tex_df = TextsDF[TextsDF.overall != 'negative'] #Drop negatives

pos_tex_df = pos_tex_df[:5000]
neg_tex_df = neg_tex_df[:5000]

print("neg.shape", neg_tex_df.shape)
print("pos.shape", pos_tex_df.shape)

Bal_DF = pd.concat([pos_tex_df, neg_tex_df], axis=0)


Bal_DF_R = Bal_DF.iloc[np.random.permutation(len(Bal_DF))]
Final_DF = Bal_DF_R.reset_index(drop=True)

print(Final_DF, Final_DF.shape)

# Splitting DataFrame in batches and Make an array of DataFrames

def split_dataframe(df, chunk_size=1000):
    list_of_df = list()
    number_chunks = len(df) // chunk_size + 1
    for i in range(number_chunks):
        list_of_df.append(df[i*chunk_size:(i+1)*chunk_size])
    return list_of_df 

Final_DF = split_dataframe(Final_DF, chunk_size=500)


def train_zero_shot_distil(batches, model_name="distilbert-base-uncased", prompt = ", this review is [MASK]."):
  
  from tqdm.notebook import tqdm
  from transformers import TFAutoModelForMaskedLM 
  from transformers import AutoTokenizer

  print("Training", model_name)
  model = TFAutoModelForMaskedLM.from_pretrained(model_name)
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  
  curRow = 0

  positive_token = tokenizer('positive', add_special_tokens=False)['input_ids'][0]
  negative_token = tokenizer('negative', add_special_tokens=False)['input_ids'][0]

  for i in tqdm(range(len(batches)), desc="Number of batches trained"):

    for _ in tqdm(range(batches[i].shape[0]), leave=False, desc="Current batch progress"):
      text = batches[i].at[curRow, 'review']
      text+=prompt

      inputs = tokenizer(text, return_tensors="tf")
      token_logits = model(**inputs).logits

      # Find the location of [MASK] and extract its logits

      mask_token_index = np.argwhere(inputs["input_ids"] == tokenizer.mask_token_id)[0, 1]

      mask_token_logits = token_logits[0, mask_token_index, :]
      # Pick the [MASK] candidates with the highest logits
      # We negate the array before argsort to get the largest, not the smallest, logits

      top_5_tokens = np.argsort(-mask_token_logits)[:30].tolist()

      if positive_token in top_5_tokens:
        batches[i].at[curRow, 'result'] = "positive"
      elif negative_token in top_5_tokens:
        batches[i].at[curRow, 'result'] = "negative"

      curRow += 1

    #Save results to CSV for later consultation
    batches[i].to_csv('/content/drive/MyDrive/Zero_shot_folds_10k/' + model_name + '_Fold' + str(i) + '.csv')

  return

def train_zero_shot_base(batches, model_name="bert-base-uncased", prompt = ", this review is [MASK]."):
  
  from tqdm.notebook import tqdm
  from transformers import TFAutoModelForMaskedLM 
  from transformers import AutoTokenizer

  print("Training", model_name)
  model = TFAutoModelForMaskedLM.from_pretrained(model_name)
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  
  curRow = 0

  positive_token = tokenizer('positive', add_special_tokens=False)['input_ids'][0]
  negative_token = tokenizer('negative', add_special_tokens=False)['input_ids'][0]

  for i in tqdm(range(len(batches)), desc="Number of batches trained"):

    for _ in tqdm(range(batches[i].shape[0]), leave=False, desc="Current batch progress"):
      text = batches[i].at[curRow, 'review']
      text+=prompt

      inputs = tokenizer(text, return_tensors="tf")
      token_logits = model(**inputs).logits

      # Find the location of [MASK] and extract its logits

      mask_token_index = np.argwhere(inputs["input_ids"] == tokenizer.mask_token_id)[0, 1]

      mask_token_logits = token_logits[0, mask_token_index, :]
      # Pick the [MASK] candidates with the highest logits
      # We negate the array before argsort to get the largest, not the smallest, logits

      top_5_tokens = np.argsort(-mask_token_logits)[:30].tolist()

      if positive_token in top_5_tokens:
        batches[i].at[curRow, 'result'] = "positive"
      elif negative_token in top_5_tokens:
        batches[i].at[curRow, 'result'] = "negative"

      curRow += 1

    #Save results to CSV for later consultation
    batches[i].to_csv('/content/drive/MyDrive/Zero_shot_folds_10k/' + model_name + '_Fold' + str(i) + '.csv')

  return

train_zero_shot_base(Final_DF)

train_zero_shot_distil(Final_DF)

def train_few_shot(batches, model_name="bert-base-uncased", prompt = ", this review is [MASK].", few_shot_initial_samples=12):
  
  from tqdm.notebook import tqdm
  from transformers import TFAutoModelForMaskedLM 
  from transformers import AutoTokenizer

  print("Training", model_name)
  model = TFAutoModelForMaskedLM.from_pretrained(model_name)
  tokenizer = AutoTokenizer.from_pretrained(model_name)

  curRow = 0

  positive_token = tokenizer('positive', add_special_tokens=False)['input_ids'][0]
  negative_token = tokenizer('negative', add_special_tokens=False)['input_ids'][0]

  for i in tqdm(range(len(batches)), desc="Number of batches trained"):
    for _ in tqdm(range(batches[i].shape[0]), leave=False, desc="Current batch progress"):
      #Generate first x samples with [MASK] uncovered
      if curRow < few_shot_initial_samples:
        text = batches[i].at[curRow, 'review']
        text += ", this review is " + str(batches[i].at[curRow, 'overall']) + "."
        batches[i].at[curRow, 'review'] = text
        batches[i].at[curRow, 'result'] = batches[i].at[curRow, 'overall']
        curRow += 1
        continue
      text = batches[i].at[curRow, 'review']
      text+=prompt
      inputs = tokenizer(text, return_tensors="tf")
      token_logits = model(**inputs).logits

      # Find the location of [MASK] and extract its logits
      mask_token_index = np.argwhere(inputs["input_ids"] == tokenizer.mask_token_id)[0, 1]
      mask_token_logits = token_logits[0, mask_token_index, :]

      # Pick the [MASK] candidates with the highest logits
      # We negate the array before argsort to get the largest, not the smallest, logits
      top_tokens = np.argsort(-mask_token_logits)[:30].tolist()
      if positive_token in top_tokens:
        batches[i].at[curRow, 'result'] = "positive"
      elif negative_token in top_tokens:
        batches[i].at[curRow, 'result'] = "negative"
      curRow += 1

    #Save results to CSV for later consultation
    batches[i].to_csv('/content/drive/MyDrive/Few_shot_folds/' + model_name + '_Fold' + str(i) + '.csv')

  return

train_few_shot(Final_DF, model_name="distilbert-base-uncased")
train_few_shot(Final_DF)

