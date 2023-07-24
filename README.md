# A Comparative Analysis of Large vs Small Language Models: Are Bigger Models Always Better?

This repository hosts a compelling study delving into the capabilities and efficiencies of Language Models (LMs). The project is a collaborative endeavor by **David Cabestany and Clara Adsuar**.

<p align="center">
  <img src="https://image.url/dataset_visualization" />
</p>

## 🚩 Table of Contents

1. [Project Synopsis](#project-synopsis)
2. [The Need for this Study](#the-need-for-this-study)
3. [Methodologies](#methodologies)
4. [Models and Tools](#models-and-tools)
5. [Learning Techniques](#learning-techniques)
6. [The Dataset](#the-dataset)
7. [Key Findings](#key-findings)
8. [What Does this Mean?](#what-does-this-mean)
9. [Getting Started](#getting-started)
10. [How to Contribute](#how-to-contribute)
11. [Peek into the Code](#peek-into-the-code)
12. [Kudos](#kudos)
13. [Reach Out](#reach-out)
14. [License](#license)

## 🧪 Project Synopsis

This study investigates the growth of Language Models (LMs) in terms of their size and popularity over the recent decades. It primarily focuses on a comparative analysis of the performance of the large-scale BERT model against the smaller DistilBERT model, both pre-trained with Masked Language Modeling and tested for sentiment classification on the same dataset.

## 💡 The Need for this Study

The race to achieve superior performances in the field of Deep Learning and NLP has cultivated a "more is better" mindset. However, potential pitfalls of large LMs like biases, harmful learning patterns, and the hefty environmental and financial costs cannot be ignored. Our project challenges this norm, and explores if smaller models like DistilBERT can match the efficiency of larger ones like BERT, while being more resource-friendly.

## 🛠️ Methodologies

- **Masked Language Modeling (MLM)**: A common pre-training method in NLP for deriving text representations.
- **Transfer Learning**: A technique that employs knowledge from one set of tasks to aid in a new, similar task.
- **Few and Zero-shot Learning**: Methods devised to enable deep learning models to generalize with minimal (or zero) example instances.

## 🧩 Models and Tools

- **BERT**: A Transformer-based model for language representation that employs unlabelled text for pre-training bidirectional representations.
- **DistilBERT**: A leaner and more efficient rendition of BERT, obtained through knowledge distillation, that displays comparable performance on numerous downstream tasks.

## 🎓 Learning Techniques

We make use of few-shot and zero-shot learning techniques in this study, as labeled datasets can be resource-intensive due to constraints like time, human effort, and financial inputs. These methods strive to develop a deep learning model that can generalize effectively with minimal (or zero) example instances.

## 📊 The Dataset

Our research makes use of the Amazon product dataset curated by Julian McAuley for sentiment analysis in text classification. We have selected the 'Movies and TV' category, extracting 10,000 reviews from the 5-core subset for this study.

<p align="center">
  <img src="https://image.url/dataset" />
</p>

## 📈 Key Findings

For an in-depth view of the results, including confusion matrices and performance metrics for few-shot and zero-shot learning for both DistilBERT and BERTBase, we encourage you to refer to the original paper. In a nutshell, despite BERTBase performing slightly better in some scenarios, DistilBERT displayed robust performance, even with a lower parameter count.

## 🌎 What Does this Mean?

Our findings pose a strong counter to the perception that larger LMs are inherently better. It establishes that leaner models like DistilBERT can compete efficiently in terms of performance while also being more resource-efficient and environmentally friendly. 

## ⚙️ Getting Started

(Provide instructions for installation and usage here)

## 🤝 How to Contribute

(Provide details about how others can contribute to your project here)

## 👀 Peek into the Code

For a better understanding of the project's inner workings, check out these [code snippets](./CODE_SNIPPETS.md).

## 👏 Kudos

We developed this project as a final assignment for the subject Deep Learning in the Master’s program HAP-LAP at the University of the Basque Country (EHU/UPV). Our supervisor for the project was Ander Barrena.

## 📞 Reach Out

- David Cabestany - dcabesma@gmail.com

We appreciate your interest in our project! For more information, please visit the project repository and feel free to reach out with any questions or comments. 

## 📜 License

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/username/repo/LICENSE.md) file for details.

---

_Last Updated: 2023-07-24_

Feel free to ⭐️ this repository if this project interests you! 😊
