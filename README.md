# Are Large Models Better than Small Models A kind approach

A fascinating exploration of the capabilities and efficiencies of Language Models (LMs) by **David Cabestany and Clara Adsuar**.

## Overview

This project investigates the effectiveness and efficiency of large-scale LMs (BERT) and their smaller versions (DistilBERT), focusing on their performance, computational cost, and environmental footprint. These LMs have gained immense popularity in recent years for their incredible performance in Deep Learning and NLP tasks.

## Why this study

With the exponential growth of LMs, a common assumption has been - the larger the model, the better the performance. However, the potential drawbacks of this "bigger is better" mindset, such as biases, harmful learning patterns, and high financial/environmental costs, are often overlooked.

We aim to shed light on this topic by comparing and analyzing the performance of BERT and DistilBERT, focusing on the principle that quality doesn't always necessarily mean larger quantities.

## Methodologies

1. **Masked Language Modeling (MLM)** - A broadly used pre-training method in NLP for learning text representations.
2. **Transfer Learning** - The technique involves transferring the knowledge from a set of tasks to a new similar task.
3. **Few and Zero-shot Learning** - Techniques to create deep learning models that learn how to generalize with a low (or null) quantity of examples.

## Tools and Models

1. **BERT** - Transformer-based model for language representation that uses unlabelled text for pre-training bidirectional representations.
2. **DistilBERT** - A smaller, more efficient version of BERT obtained through knowledge distillation that shows comparable performance on many downstream tasks.

## Dataset

Our study uses the Amazon product data created by Julian McAuley for sentiment analysis in text classification. Specifically, we chose the 'Movies and TV' category from the dataset, extracting 10,000 reviews from the 5-core subset.

![Amazon Movies and TV Shows Dataset](https://image.url/dataset)

## Results

### Few-shot DistilBERT Confusion Matrix

|           | Negative | Positive |
| --------- | -------- | -------- |
| Negative  | 4552     | 68       |
| Positive  | 3998     | 321      |

### Few-shot BERTBase Confusion Matrix

|           | Negative | Positive |
| --------- | -------- | -------- |
| Negative  | 4906     | 12       |
| Positive  | 4737     | 81       |

### Metrics Few-shot DistilBERT

|           | Value |
| --------- | ----- |
| Precision | 0.452 |
| Recall    | 0.324 |
| F1 Macro  | 0.263 |
| F1 Micro  | 0.487 |

### Metrics Few-shot BERTBase

|           | Value |
| --------- | ----- |
| Precision | 0.459 |
| Recall    | 0.332 |
| F1 Macro  | 0.233 |
| F1 Micro  | 0.498 |



## What does this mean?

The implications of this project are twofold: 

1. It challenges the notion that larger LMs are inherently superior, showing that smaller models like DistilBERT can compete effectively in terms of performance. 
2. This study also brings to the forefront the importance of considering the environmental and financial cost of training LMs. In a world where both economic and ecological factors are of paramount importance, moving towards more efficient and "greener" LMs could be a significant step in the right direction.

## Contact Information

- David Cabestany - dcabesma@gmail.com

We appreciate your interest in our project! For more information, please visit the project repository and feel free to reach out with any questions or comments. 

---

_NOTE: This project is continuously updated. The latest update was on July 20, 2023._

---

Feel free to ⭐️ this repository if this project interests you! 😊

---

## License

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/username/repo/LICENSE.md) file for details.

---

Last Updated: 2022-07-20
