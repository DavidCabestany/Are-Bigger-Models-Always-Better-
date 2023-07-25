import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from transformers import TFAutoModelForMaskedLM, AutoTokenizer
import numpy as np
import json
import gzip
from urllib.request import urlopen


class DataProcessor:
    def load_data(self, path):
        data = []
        with gzip.open(path) as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data

    def preprocess_data(self, data):
        df = pd.DataFrame.from_dict(data)
        df_filled = df.fillna('')
        reviews_df = df_filled[["summary", "overall"]]
        reviews_df.columns = ['review', 'overall']
        reviews_df['result'] = "unknown"
        reviews_df.overall = reviews_df.overall.replace({5.0: 'positive', 4.0: 'positive', 3.0: 'negative', 2.0: 'negative', 1.0: 'negative'})
        return reviews_df

    def split_dataframe(self, df, chunk_size=1000):
        chunks = []
        num_chunks = len(df) // chunk_size + 1
        for i in range(num_chunks):
            chunks.append(df[i*chunk_size:(i+1)*chunk_size])
        return chunks


class BaseModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = TFAutoModelForMaskedLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def train(self, batches, prompt, few_shot=False, few_shot_initial_samples=12):
        positive_token = self.tokenizer('positive', add_special_tokens=False)['input_ids'][0]
        negative_token = self.tokenizer('negative', add_special_tokens=False)['input_ids'][0]

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
                inputs = self.tokenizer(text, return_tensors="tf")
                token_logits = self.model(**inputs).logits
                mask_token_index = np.argwhere(inputs["input_ids"] == self.tokenizer.mask_token_id)[0, 1]
                mask_token_logits = token_logits[0, mask_token_index, :]
                top_tokens = np.argsort(-mask_token_logits)[:30].tolist()

                if positive_token in top_tokens:
                    batch.at[row_num, 'result'] = "positive"
                elif negative_token in top_tokens:
                    batch.at[row_num, 'result'] = "negative"

            batch.to_csv('/content/drive/MyDrive/Zero_shot_folds_10k/' + self.model_name + '_Fold' + str(batch_num) + '.csv')

    def evaluate(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred, labels=["positive", "negative"])
        macro = precision_recall_fscore_support(y_true, y_pred, average='macro')
        micro = precision_recall_fscore_support(y_true, y_pred, average='micro')
        acc = accuracy_score(y_true, y_pred)
        return cm, macro, micro, acc


def main():
    data_path = '/content/drive/MyDrive/Colab Notebooks/Movies_and_TV.json.gz'
    processor = DataProcessor()
    data = processor.load_data(data_path)
    reviews_df = processor.preprocess_data(data)
    batches = processor.split_dataframe(reviews_df, chunk_size=500)

    model_names = ["bert-base-uncased", "distilbert-base-uncased"]
    for model_name in model_names:
        model = BaseModel(model_name)
        model.train(batches, ", this review is [MASK].")
        model.train(batches, ", this review is [MASK].", few_shot=True)
        # model.evaluate(...)  # Call the evaluate method with the appropriate parameters

if __name__ == "__main__":
    main()
