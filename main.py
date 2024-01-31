from transformers import pipeline
import streamlit as st

for_test = 'Success'
model_ckpt = "papluca/xlm-roberta-base-language-detection"

@st.cache_resource
def load_model():
    pipe = pipeline("text-classification", model=model_ckpt)
    return pipe

model = load_model()

st.title('Определитель языка')

text = st.text_input('Введите текст')

identify = st.button('Определить')

if identify:
    out = model(text, top_k=1, truncation=True)
    st.write('Результат')
    st.success(out)
