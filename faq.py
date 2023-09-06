from os import environ
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import extract,metadata,YouTube
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import pinecone
from langchain.schema import Document
import streamlit as st
from bs4 import BeautifulSoup
import requests

# Function to extract transcript from a video file
def extract_transcript(video_filename):
    # You need to implement this function to extract the transcript using a tool like YouTube-dl or a YouTube transcript API.
    # This example uses BeautifulSoup to scrape the transcript from YouTube's web page.
    url = 'https://www.youtube.com/watch?v=' + yt.video_id
    html = requests.get(url)
    soup = BeautifulSoup(html.text, 'html.parser')
    transcript = soup.find_all('div', class_='cue-group style-scope ytd-transcript-body-renderer')
    text = ""  # Extract and process the transcript text here
    return text

# Function to generate FAQs from a transcript
def generate_faqs_from_transcript(transcript):
    # Implement your FAQ generation logic here
    # This is where you process the transcript text to generate FAQs
    # You can use NLP libraries like spaCy or NLTK to assist in FAQ generation
    # Return a list of FAQs as strings
    faqs = ["FAQ 1: Answer 1", "FAQ 2: Answer 2"]
    return faqs



# Define the Streamlit app
st.title("YouTube FAQs Generator")

# Input box for entering the YouTube URL
youtube_url = st.text_input("Enter the YouTube URL:")

# Button to generate FAQs
if st.button("Generate FAQs"):
    if youtube_url:
        try:
            # Download the video transcript
            yt = YouTube(youtube_url)
            st.write(f"Downloading video transcript for '{yt.title}'...")
            transcript = extract_transcript(yt.title + ".mp4")

            # Process the transcript to generate FAQs (you need to implement this function)
            faqs = generate_faqs_from_transcript(transcript)

            # Display the FAQs
            st.subheader("Generated FAQs:")
            for i, faq in enumerate(faqs, 1):
                st.write(f"{i}. {faq}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a YouTube URL.")

