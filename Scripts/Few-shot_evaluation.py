from google.colab import drive
drive.mount('/content/drive')

import pandas as pd

TextsDF_Base = []
TextsDF_Distil = []

for i in range(21):
  TextsDF_Base.append(pd.read_csv('/content/drive/MyDrive/Few_shot_folds/bert-base-uncased_Fold'+str(i)+'.csv'))
  TextsDF_Distil.append(pd.read_csv('/content/drive/MyDrive/Few_shot_folds/distilbert-base-uncased_Fold'+str(i)+'.csv'))

from sklearn.metrics import confusion_matrix

y_true_base = []
y_true_distil = []
y_predict_base = []
y_predict_distil = []
for b in range(21):
  y_true_distil.extend(TextsDF_Distil[b]['overall'].to_list())
  y_predict_distil.extend(TextsDF_Distil[b]['result'].to_list())

  y_true_base.extend(TextsDF_Base[b]['overall'].to_list())
  y_predict_base.extend(TextsDF_Base[b]['result'].to_list())

cm_distil = confusion_matrix(y_true_distil, y_predict_distil, labels=["positive", "negative"])
cm_base = confusion_matrix(y_true_base, y_predict_base, labels=["positive", "negative"])

print("Confusion matrix for DistilBERT_12_shot")
print(cm_distil)
print("Confusion matrix for BERT-Base_12_shot")
print(cm_base)

from sklearn.metrics import precision_recall_fscore_support

precision_recall_fscore_support_MACRO_distil = precision_recall_fscore_support(y_true_distil, y_predict_distil, average='macro')
precision_recall_fscore_support_MACRO_base = precision_recall_fscore_support(y_true_base, y_predict_base, average='macro')

precision_recall_fscore_support_MICRO_distil = precision_recall_fscore_support(y_true_distil, y_predict_distil, average='micro')
precision_recall_fscore_support_MICRO_base = precision_recall_fscore_support(y_true_base, y_predict_base, average='micro')

print("Negative Precision, recall, F1 MACRO, Support for DistilBERT:")
print(precision_recall_fscore_support_MACRO_distil)

print("Negative Precision, recall, F1 MACRO, Support for BERT-Base:")
print(precision_recall_fscore_support_MACRO_base)

print("Positive Precision, recall, F1 MICRO, Support for DistilBERT:")
print(precision_recall_fscore_support_MICRO_distil)

print("Positive Precision, recall, F1 MICRO, Support for BERT-Base:")
print(precision_recall_fscore_support_MICRO_base)

from sklearn.metrics import accuracy_score

acc_Distil = accuracy_score(y_true_distil, y_predict_distil)
acc_Base = accuracy_score(y_true_base, y_predict_base)

print("Accuracy for DistilBERT_zero_shot")
print(acc_Distil)
print("Accuracy for BERT-Base_zero_shot")
print(acc_Base)

