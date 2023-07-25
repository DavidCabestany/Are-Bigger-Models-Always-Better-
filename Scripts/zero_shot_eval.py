import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

# Load the data
base_data = []
distil_data = []

for i in range(21):
    base_data.append(pd.read_csv('/content/drive/MyDrive/Zero_shot_folds_10k/bert-base-uncased_Fold' + str(i) + '.csv'))
    distil_data.append(pd.read_csv('/content/drive/MyDrive/Zero_shot_folds_10k/distilbert-base-uncased_Fold' + str(i) + '.csv'))

# Generate confusion matrix
true_base = []
true_distil = []
predict_base = []
predict_distil = []

for b in range(21):
    true_distil.extend(distil_data[b]['overall'].to_list())
    predict_distil.extend(distil_data[b]['result'].to_list())

    true_base.extend(base_data[b]['overall'].to_list())
    predict_base.extend(base_data[b]['result'].to_list())

cm_distil = confusion_matrix(true_distil, predict_distil, labels=["positive", "negative"])
cm_base = confusion_matrix(true_base, predict_base, labels=["positive", "negative"])

print("Confusion matrix for DistilBERT")
print(cm_distil)
print("Confusion matrix for BERT-Base")
print(cm_base)

# Precision, recall, F-score, support
macro_distil = precision_recall_fscore_support(true_distil, predict_distil, average='macro')
macro_base = precision_recall_fscore_support(true_base, predict_base, average='macro')

micro_distil = precision_recall_fscore_support(true_distil, predict_distil, average='micro')
micro_base = precision_recall_fscore_support(true_base, predict_base, average='micro')

print("Precision, recall, F1 MACRO, Support for DistilBERT:")
print(macro_distil)

print("Precision, recall, F1 MACRO, Support for BERT-Base:")
print(macro_base)

print("Precision, recall, F1 MICRO, Support for DistilBERT:")
print(micro_distil)

print("Precision, recall, F1 MICRO, Support for BERT-Base:")
print(micro_base)

# Accuracy
acc_distil = accuracy_score(true_distil, predict_distil)
acc_base = accuracy_score(true_base, predict_base)

print("Accuracy for DistilBERT_zero_shot")
print(acc_distil)
print("Accuracy for BERT-Base_zero_shot")
print(acc_base)
