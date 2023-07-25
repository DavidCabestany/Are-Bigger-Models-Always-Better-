import pandas as pd
from tqdm import tqdm
from transformers import TFAutoModelForMaskedLM, AutoTokenizer
import numpy as np
import json
import gzip
from urllib.request import urlopen

def load_data(path):
    """Load data from a gzipped JSON file."""
    data = []
    with gzip.open(path) as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def prepare_data(data):
    """Prepare data for training."""
    df = pd.DataFrame.from_dict(data)
    df_filled = df.fillna('')
    reviews_df = df_filled[["summary", "overall"]]
    reviews_df.columns = ['review', 'overall']
    reviews_df['result'] = "unknown"
    reviews_df.overall = reviews_df.overall.replace({5.0: 'positive', 4.0: 'positive', 3.0: 'negative', 2.0: 'negative', 1.0: 'negative'})
    return reviews_df

def split_dataframe(df, chunk_size=1000):
    """Split a DataFrame into smaller chunks."""
    chunks = []
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks

def train_model(batches, model_name, prompt, few_shot=False, few_shot_initial_samples=12):
    """Train the model and save the results."""
    model = TFAutoModelForMaskedLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    positive_token = tokenizer('positive', add_special_tokens=False)['input_ids'][0]
    negative_token = tokenizer('negative', add_special_tokens=False)['input_ids'][0]

    for batch_num, batch in enumerate(batches):
        for row_num in range(batch.shape[0]):
            if few_shot and row_num < few_shot_initial_samples:
                text = batch.at[row_num, 'review']
                text += ", this review is " + str(batch.at[row_num, 'overall']) + "."
                batch.at[row_num, 'review'] = text
                batch.at[row_num, 'result'] = batch.at[row_num, 'overall']
                continue

            text = batch.at[row_num, 'review']
            text += prompt
            inputs = tokenizer(text, return_tensors="tf")
            token_logits = model(**inputs).logits
            mask_token_index = np.argwhere(inputs["input_ids"] == tokenizer.mask_token_id)[0, 1]
            mask_token_logits = token_logits[0, mask_token_index, :]
            top_tokens = np.argsort(-mask_token_logits)[:30].tolist()

            if positive_token in top_tokens:
                batch.at[row_num, 'result'] = "positive"
            elif negative_token in top_tokens:
                batch.at[row_num, 'result'] = "negative"

        batch.to_csv('/content/drive/MyDrive/Zero_shot_folds_10k/' + model_name + '_Fold' + str(batch_num) + '.csv')

def main():
    data_path = '/content/drive/MyDrive/Colab Notebooks/Movies_and_TV.json.gz'
    data = load_data(data_path)
    reviews_df = prepare_data(data)
    batches = split_dataframe(reviews_df, chunk_size=500)

    train_model(batches, model_name="bert-base-uncased", prompt=", this review is [MASK].")
    train_model(batches, model_name="distilbert-base-uncased", prompt=", this review is [MASK].")
    train_model(batches, model_name="distilbert-base-uncased", prompt=", this review is [MASK].", few_shot=True)
    train_model(batches, model_name="bert-base-uncased", prompt=", this review is [MASK].", few_shot=True)

if __name__ == "__main__":
    main()
