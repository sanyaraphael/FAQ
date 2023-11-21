from os import environ
from langchain.llms import OpenAI, HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
# from transformers import LLMChain, OutputParser
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import extract,metadata,YouTube
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import pinecone
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
import streamlit as st
from bs4 import BeautifulSoup
import requests
import pandas as pd
from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings

load_dotenv()
openai_api_key=environ.get("OPEN_AI_API_KEY")
pinecone_api_key=environ.get("PINECONE_API_KEY")
project_name=environ.get("PROJECT_NAME")
pinecone_env=environ.get("PINECONE_ENVIRONMENT")
index_name=environ.get("INDEX_NAME")
huggingfaceAPIkey=environ.get("HUGGINGFACEHUB_API_TOKEN")


def get_text_chunks(transcript) :
    transcript=str(transcript)
    docs=[]
    text_splitter = CharacterTextSplitter(separator=" ", chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(transcript)
    for chunk in chunks:
        docs.append(Document(page_content=chunk))
    return docs

def generate_faqs_from_transcript(transcript,youtube_url):
   
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    # embeddings=OpenAIEmbeddings(openai_api_key=openai_api_key)

    pinecone.init(      
	api_key=pinecone_api_key,      
	environment=pinecone_env      
    )      
    index = pinecone.Index('faq')

    docsearch=Pinecone.from_documents([],embeddings,index_name=index_name)

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

        # docs=get_text_chunks(transcript)
        # print(type(docs))
        # if docs is not None:
        #  docsearch = Pinecone.from_documents(documents=docs,embedding=embeddings,index_name=index_name)
        # docs = list(map(str, docs))

        # st.write(docs)
        # docsearch=Pinecone.from_texts(texts=docs,embedding=embeddings)
        


        #storing docs to pinecone
        st.info(len(docs))




        # # Initialize LLMChain and OutputParser
        # model = LLMChain.from_pretrained("model_name")
        # output_parser = OutputParser()

        # for doc in docs:
        #     # Generate text and parse the output
        #     output = model(doc, output_parser=output_parser)

        #     # Access the parsed output using output
        #     parsed_output = output_parser(output)
        #     st.write(parsed_output)

    # with st.spinner("Adding data to the vector store"):
        # docsearch = Pinecone.from_documents(documents=docs,embedding=embeddings,index_name=index_name)

    # user_question = st.text_input("Ask a question from the uploaded videos",placeholder="Type your question here")
    user_question="Generate 10 questions using the data"
        
    # if user_question:

    # llm=HuggingFaceHub(repo_id="tiiuae/falcon-180B",model_kwargs={"temperature":0.5,"max_length":512},huggingfacehub_api_token=huggingfaceAPIkey)
    # llm=HuggingFaceHub(repo_id="meta-llama/Llama-2-7b-chat",model_kwargs={"temperature":0.5,"max_length":512},huggingfacehub_api_token=huggingfaceAPIkey)
    # llm=HuggingFaceHub(repo_id="google/flan-t5-xxl",model_kwargs={"temperature":0.5,"max_length":512},huggingfacehub_api_token=huggingfaceAPIkey)
    # llm=HuggingFaceHub(repo_id="deepset/roberta-base-squad2",model_kwargs={"temperature":0.5,"max_length":512},huggingfacehub_api_token=huggingfaceAPIkey)

    #load question answer from langchain library
    # chain = load_qa_chain(llm, chain_type="stuff")
    # for doc in docs:
    #     response = chain.run(context=doc, question=user_question,input_documents=docs)
    #     st.write(response)

    
 

    # model = "tiiuae/falcon-180b"
    model = "meta-llama/Llama-2-13b-chat-hf"
    model = AutoModelForCausalLM.from_pretrained(model)

    tokenizer = AutoTokenizer.from_pretrained(model)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    for texts in text_array:
        sequences = pipeline(
        texts,   max_length=200,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
        )
        for seq in sequences:
            print(f"Result: {seq['generated_text']}")


    # # Load the fine-tuned question generation model
    # model_name = "tiiuae/falcon-180B"
    # # model_name="potsawee/t5-large-generation-squad-QuestionAnswer"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(model_name)

 
    # for texts in text_array:
    #     # Generate questions from the input text
    #     input_ids = tokenizer.encode("generate questions: " + texts, return_tensors="pt", max_length=1024, truncation=True)
    #     output_ids = model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

    #     # Decode and print the generated questions
    #     generated_questions = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    #     print("Generated Questions:")
    #     print(generated_questions)
    #     st.write(generated_questions)

    
    # get_openai_callback() is used to check the billing info of openai
    # with get_openai_callback() as cb:
    #     response = chain.run(input_documents=docs, question=user_question)
    #     print(cb)
    
    with st.spinner("Generating FAQs"):       
        faqs = ["FAQ 1: Answer 1", "FAQ 2: Answer 2"]
        return faqs

def getCoversationChain(vectorstore):
    llm=HuggingFaceHub(repo_id="meta-llama/Llama-2-7b-chat",model_kwargs={"temperature":0.5,"max_length":512})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever (), memory=memory)
    return conversation_chain


def main():

    # Add custom CSS to set the background color to white
    st.markdown(
        """
        <style>
        body {
            background-color: #ffffff; /* White background color */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # Define the Streamlit app
    st.title("YouTube FAQs Generator")

    # Input box for entering the YouTube URL
    youtube_url = st.text_input("Enter the YouTube URL:")

    # Button to generate FAQs
    if st.button("Generate FAQs"):
        if youtube_url:
            # try:
                # st.write(f"Downloading video transcript for '{yt.title}'...")
                with st.spinner("Downloading video transcript"):
                    # transcript=download_youtube_transcript(youtube_url)
                    video_id=extract.video_id(youtube_url)
                    transcript = YouTubeTranscriptApi.get_transcript(video_id)
        
                # st.write(transcript)

                # Process the transcript to generate FAQs (you need to implement this function)
                faqs = generate_faqs_from_transcript(transcript,youtube_url)

                # Display the FAQs
                st.subheader("Generated FAQs:")
                for i, faq in enumerate(faqs, 1):
                    st.write(f"{i}. {faq}")
            # except Exception as e:
            #     st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a YouTube URL.")
    

def download_youtube_transcript(video_url):
    try:
        # Get the transcript for the YouTube video
        video_id=extract.video_id(video_url)
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        
        # Combine the transcript into a single string
        transcript_text = ""
        for entry in transcript:
            transcript_text += entry['text']
        
        # Write the transcript to a text file
        with open("transcript.txt", "w", encoding="utf-8") as file:
            file.write(transcript_text)
        
    except Exception as e:
        print("An error occurred:", str(e))

    return transcript_text

    

if __name__=='__main__':
    main()

