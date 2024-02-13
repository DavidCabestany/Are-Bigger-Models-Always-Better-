from google.colab import drive
drive.mount('/content/drive')

!pip install transformers

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

#!wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Home_and_Kitchen_5.json.gz


### load the meta data

data = []
with gzip.open('/content/drive/MyDrive/Colab Notebooks/Movies_and_TV.json.gz') as f:
    for l in f:
        data.append(json.loads(l.strip()))
    
# total length of list, this number equals total number of products
#print(len(data))

# first row of the list
#print(data[0:5])

# convert list into pandas dataframe

df = pd.DataFrame.from_dict(data)

dataframe = df

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

neg_tex_df = TextsDF[TextsDF.overall != 'positive']
pos_tex_df = TextsDF[TextsDF.overall != 'negative']

pos_tex_df = pos_tex_df[:5000]
neg_tex_df = neg_tex_df[:5000]

print("neg.shape", neg_tex_df.shape)
print("pos.shape", pos_tex_df.shape)


Bal_DF = pd.concat([pos_tex_df, neg_tex_df], axis=0)


Bal_DF_R = Bal_DF.iloc[np.random.permutation(len(Bal_DF))]
Final_DF = Bal_DF_R.reset_index(drop=True)

print(Final_DF, Final_DF.shape)

def split_dataframe(df, chunk_size=1000):
    list_of_df = list()
    number_chunks = len(df) // chunk_size + 1
    for i in range(number_chunks):
        list_of_df.append(df[i*chunk_size:(i+1)*chunk_size])
    return list_of_df 

Final_DF = split_dataframe(Final_DF, chunk_size=500)



#print(token_dbe('positive', add_special_tokens=False)['input_ids'][0])

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

      top_5_tokens = np.argsort(-mask_token_logits)[:30].tolist()

      if positive_token in top_5_tokens:
        batches[i].at[curRow, 'result'] = "positive"
      elif negative_token in top_5_tokens:
        batches[i].at[curRow, 'result'] = "negative"

      curRow += 1

    #Save results to CSV for later consultation
    batches[i].to_csv('/content/drive/MyDrive/Few_shot_folds/' + model_name + '_Fold' + str(i) + '.csv')

  return

def train_few_shot(batches, model_name="bert-base-uncased", prompt = ", this review is [MASK].", few_shot_initial_samples=12):
  
  from tqdm.notebook import tqdm
  from transformers import TFAutoModelForMaskedLM 
  from transformers import AutoTokenizer

  print("Training", model_name)
  model = TFAutoModelForMaskedLM.from_pretrained(model_name)
  tokenizer = AutoTokenizer.from_pretrained(model_name)

  #for s in range(12):
  #  text = batches[0].at[curRow, 'review']
   # fewshot_prompt = ", this review is " + 
  
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

      top_5_tokens = np.argsort(-mask_token_logits)[:30].tolist()

      if positive_token in top_5_tokens:
        batches[i].at[curRow, 'result'] = "positive"
      elif negative_token in top_5_tokens:
        batches[i].at[curRow, 'result'] = "negative"

      curRow += 1

    #Save results to CSV for later consultation
    batches[i].to_csv('/content/drive/MyDrive/Few_shot_folds/' + model_name + '_Fold' + str(i) + '.csv')

  return

train_few_shot(Final_DF, model_name="distilbert-base-uncased")
train_few_shot(Final_DF)


#train_zero_shot_distil(TextsDF)

print("Positive results: ", TextsDF[0].groupby('result').get_group('positive').shape[0])
print("Negative results: ", TextsDF[0].groupby('result').get_group('negative').shape[0])
print("Unclassified results: ", TextsDF[0].groupby('result').get_group('unknown').shape[0])

########################

# FEW-SHOT DISTILLBERT #

########################

one = "This is a really good movie, this review is positive."
two = "This is a really bad movie, this review is negative."
three = "This mug does only a fair job of keeping coffee hot.  It will absolutely NOT protect your shirt, pants, carseat, or anything else from spills and drips!  The lid does help to keep the coffee hot in that without the lid it is exposed to the air.  That slight benefit and it's attractive appearance are the only advantages I found.  If you have NO concern about leaking/dripping, this mug is sufficient, however, the Contigo Autoseal West Loop is a much better product, and for essentially the same price, this review is [MASK]."
text = one+" "+two+" "+three
inputs = token_dbe(text, return_tensors="np")
token_logits = model_dbe(**inputs).logits
# Find the location of [MASK] and extract its logits
mask_token_index = np.argwhere(inputs["input_ids"] == token_dbe.mask_token_id)[0, 1]
mask_token_logits = token_logits[0, mask_token_index, :]
# Pick the [MASK] candidates with the highest logits
# We negate the array before argsort to get the largest, not the smallest, logits
top_5_tokens = np.argsort(-mask_token_logits)[:5].tolist()

for token in top_5_tokens:
  """Distill BERT Positive ID 3893, Negative ID 4997"""
  if token == 3893:
    print("Distill_BERT few-shot",f">>> {text.replace(token_dbe.mask_token, token_dbe.decode([token]))}") 
    break
  if token_dbe.decode([token]) == "negative":
    print("Distill_BERT few-shot",f">>> {text.replace(token_dbe.mask_token, token_dbe.decode([token]))}") 
    break
  else:
    print("deprecated Distill_BERT few-shot",f">>> {text.replace(token_dbe.mask_token, token_dbe.decode([token]))}")
    continue
    

def train_few_shot_base(batches, model_name="bert-base-uncased", prompt = ", this review is [MASK]."):
  
  from tqdm.notebook import tqdm
  from transformers import TFAutoModelForMaskedLM 
  from transformers import AutoTokenizer

  def build_first_samples(samplesNum):
    
    from tqdm.notebook import tqdm
    
    for i in range(samplesNum):


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
    batches[i].to_csv('/content/drive/MyDrive/Zero_shot_folds/' + model_name + '_Fold' + str(i) + '.csv')

  return

import torch

label = ['positive', 'negative']
device = torch.device('cpu')    #WHY? You wanna sit here for the rest of eternity?
for _, row in TextsDF.iterrows():
  premise = str(row['review'])
  hypothesis = f'This example is {label}.'

    # run through model pre-trained on MNLI
  x = tokenizer.encode(premise, hypothesis, return_tensors='pt',    #"pt" or "tf" or "np"?
                        truncation_strategy='only_first')
  logits = nli_model(x.to(device))[0]

    # we throw away "neutral" (dim 1) and take the probability of
    # "entailment" (2) as the probability of the label being true 
  entail_contradiction_logits = logits[:,[0,2]]
  probs = entail_contradiction_logits.softmax(dim=1)
  prob_label_is_true = probs[:,1]
  print(prob_label_is_true)


########################

#ZERO-SHOT DISTILLBERT #

########################

from tqdm.notebook import tqdm


from transformers import TFAutoModelForMaskedLM 
from transformers import AutoTokenizer
import numpy as np


model_dbe = TFAutoModelForMaskedLM.from_pretrained("distilbert-base-uncased")

token_dbe = AutoTokenizer.from_pretrained("distilbert-base-uncased")

#model_dbe(model_dbe.dummy_inputs)  # Build the model
#model_dbe.summary()

promt= ", this review is [MASK]."
curRow = 0

for i in tqdm(range(len(TextsDF)), desc="Number of batches trained"):
  #print("Training batch "+str(i))

  for _ in tqdm(range(TextsDF[i].shape[0]), leave=False, desc="Current batch progress"):
    text = TextsDF[i].at[curRow, 'review']
    text+=promt

    inputs = token_dbe(text, return_tensors="tf")
    token_logits = model_dbe(**inputs).logits

    # Find the location of [MASK] and extract its logits

    mask_token_index = np.argwhere(inputs["input_ids"] == token_dbe.mask_token_id)[0, 1]

    mask_token_logits = token_logits[0, mask_token_index, :]
    # Pick the [MASK] candidates with the highest logits
    # We negate the array before argsort to get the largest, not the smallest, logits

    top_5_tokens = np.argsort(-mask_token_logits)[:30].tolist()

    if 3893 in top_5_tokens:
      TextsDF[i].at[curRow, 'result'] = "positive"
    elif 4997 in top_5_tokens:
      TextsDF[i].at[curRow, 'result'] = "negative"

    curRow += 1

  #Save results to CSV for later consultation
  TextsDF[i].to_csv('/content/drive/MyDrive/Colab Notebooks/Zero-Shot_Distilbert_Fold' + str(i) + '.csv')


