# Email_spam_classification_Using_BERT ( Context-Aware Email Spam Classification)

[![🚀 Live Streamlit App](https://img.shields.io/badge/Live%20Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://emailspamclassification-gsd8ezfi4be52ugewwfh3f.streamlit.app/)
&nbsp;&nbsp;

---

## Problem Statement

Traditional email spam classification systems rely primarily on keyword frequency, bag-of-words, or shallow embedding techniques.
While these approaches work well for detecting obvious spam patterns, they fail to capture **semantic context and user intent**, particularly in short or conversational messages.

As a result, legitimate emails such as:
“How are you?”
or even single-word inputs like:
“How”

may be incorrectly classified as spam due to the absence of strong keyword signals.

At the same time, messages that combine casual conversation with spam intent, such as:
“How are you Sree, you won the lottery”

are not consistently evaluated when models lack the ability to understand contextual relationships across the full sentence.

These limitations lead to:

* False positives where genuine (HAM) messages are flagged as spam
* Unstable predictions for short or informal emails
* Poor handling of mixed-intent messages containing both benign and malicious cues

This project addresses these issues by using a **context-aware language model (BERT)** that evaluates messages holistically rather than relying on isolated tokens.

----

## Project Architecture

```
project-root/
├── tokenizer_2/
│   └── Tokenizer files and artifacts used for text preprocessing
│
├── app.py
│   └── Application entry point for running the model or interface
│
├── model.py
│   └── Core model definition and loading logic
│
├── config.json
│   └── Model and application configuration parameters
│
├── tokenizer_config.json
│   └── Tokenizer configuration settings
│
├── special_tokens_map.json
│   └── Mapping of special tokens used by the tokenizer
│
├── vocab.txt
│   └── Vocabulary file used during tokenization
│
├── requirements.txt
│   └── Python dependencies required to run the project
│
├── README.md
│   └── Project documentation and usage details
│
└── LICENSE
    └── Project license information
```

## System Architecture

The spam classification system is designed to evaluate emails **holistically**, focusing on semantic meaning rather than isolated keywords.


**System Flow**

```
Input Email Text
        ↓
Text Cleaning & Normalization
        ↓
BERT Tokenization (WordPiece)
        ↓
Contextual Embedding (Bidirectional Attention)
        ↓
Classification Head (Spam / Ham)
        ↓
Final Prediction
```

**Key Design Choices**

* Uses a pretrained BERT model to capture bidirectional context
* Processes the full sentence jointly instead of token-level scoring
* Enables robust intent detection for short, conversational, and mixed-content messages
* Reduces false positives caused by sparse or ambiguous inputs

---

## Evaluation Examples (Why Context Matters)

The following examples highlight the limitations of classical models and the improvement achieved using BERT.

| Input Message                           | Classical Model Output | BERT-Based Output |
| --------------------------------------- | ---------------------- | ----------------- |
| `How are you?`                          | Spam                   | Ham               |
| `How`                                   | Spam                   | Ham               |
| `Are you free today?`                   | Spam                   | Ham               |
| `How are you Sree, you won the lottery` | Inconsistent           | Spam              |
| `Congratulations! You won a prize`      | Spam                   | Spam              |

**Observation**

* Classical models misclassify short or conversational messages due to weak keyword signals
* BERT correctly interprets intent by leveraging surrounding context
* Mixed-intent messages are evaluated more consistently

---

## Outcome

By incorporating a context-aware language model, the system:

* Significantly reduces false positives on genuine emails
* Improves robustness for short and informal messages
* Accurately detects spam intent embedded within normal conversation

This demonstrates the importance of **semantic understanding over keyword frequency** in real-world spam classification systems.

---





