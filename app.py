# ----------------------------------
# FIX FOR WINDOWS + TORCH OPENMP
# ----------------------------------
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import torch
from transformers import BertTokenizer
from model import SentimentClassifier

# ----------------------------------
# Page configuration
# ----------------------------------
st.set_page_config(
    page_title="Spam Detection",
    page_icon="📩",
    layout="centered"
)

# ----------------------------------
# Header
# ----------------------------------
st.title("Spam Detection using BERT")
st.caption("Classify text messages as Spam or Ham using a fine-tuned BERT model.")

# ----------------------------------
# Load tokenizer (LOCAL)
# ----------------------------------
@st.cache_resource
def load_tokenizer():
    return BertTokenizer.from_pretrained("./tokenizer_2")

# ----------------------------------
# Load model (LOCAL)
# ----------------------------------
@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model = SentimentClassifier()
    model.load_state_dict(
        torch.load("spam_bert_model_2.pt", map_location=device)
    )
    model.eval()
    return model

tokenizer = load_tokenizer()
model = load_model()

# ----------------------------------
# Input
# ----------------------------------
text = st.text_area(
    "Message",
    placeholder="Enter the message to be classified",
    height=120
)

st.write("")

# ----------------------------------
# Action
# ----------------------------------
predict = st.button("Classify", use_container_width=True)

# ----------------------------------
# Prediction
# ----------------------------------
if predict:
    if text.strip() == "":
        st.warning("Please enter a message.")
    else:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            logits = model(inputs["input_ids"], inputs["attention_mask"])
            prob = torch.sigmoid(logits).item()

        if prob >= 0.6:
            st.error("Spam message detected")
        else:
            st.success("Message classified as Ham")


