# -*- coding: utf-8 -*-
"""FAQfromVideo.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Nb-UHOcQp3-ZFS78W0QGxb99tRbMaHcl
"""

!pip install transformers torch accelerate

!huggingface-cli login

!pip install streamlit
!pip install youtube_transcript_api
!pip install pytube

!pip install HuggingFace

!pip install langchain

from transformers import AutoTokenizer
import transformers
from transformers import pipeline
import torch
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import extract,metadata,YouTube
from langchain.embeddings import HuggingFaceInstructEmbeddings

model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

sequences = pipeline(
    """look down after
lockdownthough core would cause mass destructionit increased the mass of an individualperson byincreasing the belly size if you
take apicture from spaceit looks like as if the earth ispregnant and sunis throwing a baby shiver by sayingcongratulationsyoumy
friend sarah no kumar is like thankgod the virus is gone there is no bigproblem i said the bigger problem is thesize of your bellyit
looks as if it is a ticking time bombready to beexploded in three minutes the onlyproblem is to defuse the bomb we cannotthrow just the
belly portion alone intothe wellwe need to throw the whole body how doyou know that. Generate sensible questions out of the above context""",
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=500,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")

def generate_faqs_from_transcript(transcript,youtube_url):

    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    # embeddings=OpenAIEmbeddings(openai_api_key=openai_api_key)

    thumbnail_url=YouTube(youtube_url).thumbnail_url

    text_array=[]
    docs=[]
    texts=' '
    start=0
    with st.spinner("Dividing data into chunks"):
        for i in transcript:
            if len(texts) >= 500:
                docs.append(Document(page_content=texts,metadata={"start":start,"youtube_link":youtube_url, "thumbnail_url": thumbnail_url}))
                # docs.append(page_content="hello",metadata={"start":start,"youtube_link":youtube_url, "thumbnail_url": thumbnail_url})
                text_array.append(texts)
                metadata={"start":start,"youtube_link":youtube_url, "thumbnail_url": thumbnail_url}
                texts=' '
            else:
                # print(i['start'])
                if texts ==' ':
                    start= int(i['start'])
                texts =texts+i['text']



        #storing docs to pinecone
        st.info(len(docs))

# Commented out IPython magic to ensure Python compatibility.
# %%writefile faq_from_video.py
# def main():
# 
#     # Add custom CSS to set the background color to white
#     st.markdown(
#         """
#         <style>
#         body {
#             background-color: #ffffff; /* White background color */
#         }
#         </style>
#         """,
#         unsafe_allow_html=True,
#     )
#     # Define the Streamlit app
#     st.title("YouTube FAQs Generator")
# 
#     # Input box for entering the YouTube URL
#     youtube_url = st.text_input("Enter the YouTube URL:")
# 
#     # Button to generate FAQs
#     if st.button("Generate FAQs"):
#         if youtube_url:
#             try:
#                 with st.spinner("Downloading video transcript"):
#                     video_id=extract.video_id(youtube_url)
#                     transcript = YouTubeTranscriptApi.get_transcript(video_id)
#             except Exception as e:
#                 st.error(f"An error occurred: {str(e)}")
# 
# 
#                 # Process the transcript to generate FAQs (you need to implement this function)
#                 faqs = generate_faqs_from_transcript(transcript,youtube_url)
# 
#                 # Display the FAQs
#                 st.subheader("Generated FAQs:")
#                 for i, faq in enumerate(faqs, 1):
#                     st.write(f"{i}. {faq}")
#             # except Exception as e:
#             #     st.error(f"An error occurred: {str(e)}")
#         else:
#             st.warning("Please enter a YouTube URL.")
# 
#

!npm install localtunnel

!streamlit run faq_from_video.py &>/content/logs.txt &

!npx localtunnel --port 8501