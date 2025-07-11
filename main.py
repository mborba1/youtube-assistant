import streamlit as st
import langchain_helper as lch
import textwrap
import os

st.set_page_config(
    page_title="YouTube Video Q&A",
    page_icon=":video_camera:",
    layout="wide"
)

with st.sidebar:
    st.title("YouTube Video Q&A Assistant")
    with st.form(key='my_form'):
        youtube_url = st.text_input(
            "What is the YouTube video URL?",
            placeholder="https://www.youtube.com/watch?v=...",
            max_chars=100
            )
        query = st.text_area(
            label="Ask me about the video?",
            max_chars=200,
            )
        openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password"
            )
        st.markdown("[Get an OpenAI API key](https://platform.openai.com/account/api-keys)")
        submit_button = st.form_submit_button(label='Submit')

if submit_button:
    if not openai_api_key:
        st.error("Please add your OpenAI API key to continue.")
        st.stop()
    if not youtube_url or not query:
        st.warning("Please provide both a YouTube video URL and a query.")
        st.stop()
    st.video(youtube_url)

    with st.spinner("Processing the video..."):
        try:
            db = lch.create_vector_db_from_youtube(youtube_url, openai_api_key)
            response, docs = lch.get_response_from_query(db, query, open_api_key=openai_api_key)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()
        st.success("Video processed successfully!")        
        st.subheader("Answer:")
        st.write(textwrap.fill(response, width=85))

        with st.expander ("View relevent transcript sections"):
            for doc in docs:
                st.markdown(f"-- {doc.page_content.strip()[:300]}... --")
                