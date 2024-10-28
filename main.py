import streamlit as st
import langchain_helper as lch
import textwrap

st.title("Your Document Analyzer")

with st.sidebar:
    with st.form(key='my_form'):

        query = st.text_area(
            label="Ask me about the content of the document",
            max_chars=100,
            key="query",
        )

        submit_button = st.form_submit_button(label="Submit")

if query:
    db = lch.get_data_from_file();
    response, docs = lch.get_response_from_query(db,query)

    st.subheader("Answer:")
    st.text(textwrap.fill(response, width=80))