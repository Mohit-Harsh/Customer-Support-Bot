import pandas as pd
import numpy as np
import transformers
import os
from langchain_openai import OpenAI
import pandas as pd
import numpy as np
import json
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import streamlit as st
from keybert import KeyBERT
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

@st.cache_resource
def load_resources():
    df = pd.read_csv('air_india.csv')
    df['price'] = [str(x).strip() for x in df['price'].tolist()]
    df['flightNumber'] = [str(x).strip() for x in df['flightNumber'].tolist()]

    with open(os.environ['OPENAI_API_KEY'], 'r') as file:
        api_key = file.read()

    llm = OpenAI(api_key=api_key, max_tokens=512, model_name="gpt-3.5-turbo-instruct", temperature=0, streaming=True)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    docstore = FAISS.load_local('solutions', embeddings)

    extractor = KeyBERT()

    fn = set()

    for x in df['flightNumber'].tolist():

        fn.add(x.lower().strip())

    fn = list(fn)

    return df,docstore,llm,extractor,fn

if "flight" not in st.session_state:
    st.session_state.flight=None
if "origin" not in st.session_state:
    st.session_state.origin=None
if "destination" not in st.session_state:
    st.session_state.destination=None


df,docstore,llm,extractor,fn = load_resources()

def setbooking():

    book_flight(st.session_state.origin,st.session_state.destination)

def setflight(query):

    enquiry(query + " flight number - " + st.session_state.flight,llm,extractor,fn)

def setfaq(query):

    faq(query,docstore,llm)

def book_flight(origin,destination):

    st.session_state.booking = False
    data = df[df['origin'] == origin.lower().capitalize()]
    data = data[data['destination'] == destination.lower().capitalize()]
    data = data.drop_duplicates(subset=['flightNumber']).reset_index(drop=['index'])

    with st.chat_message("assistant"):
        st.write(f"Here are the flights available from {origin} to {destination}")
        
        for i in range(len(data)):
        
            st.dataframe([data.iloc[i]])


def enquiry(query,llm,extractor,fn):

    keywords = [x[0] for x in extractor.extract_keywords(query,candidates=fn)]

    if(len(keywords) == 0):
        
        with st.form("flightno"):

            flightno = st.text_input(label='Flight Number',placeholder='Enter Flight Number',key='flight')
            st.form_submit_button("Submit",on_click=setflight, args=[query])

    else:

        for key in keywords:

            temp = df[df['flightNumber'] == key].head(1)

            with st.chat_message("assistant"):
            
                if(len(temp.values) == 0):
                    
                    st.write('Sorry! No flights are available with that Flight Number. Please enter the correct Flight Number.')
                    with st.form("flightno"):

                        flightno = st.text_input(label='Flight Number',placeholder='Enter Flight Number',key='flight')
                        st.form_submit_button("Submit",on_click=setflight, args=[query])
                    
                else:
                    
                    # context = f'''Flight Number : {temp['flightNumber'].tolist()[0]}\nArrival : {temp['scheduledArrivalTime'].tolist()[0]}\nDeparture : {temp['scheduledDepartureTime'].tolist()[0]}'''
                    st.write("Here are your flight details")
                    st.dataframe(temp)



def faq(query,docstore,llm):

    docs = docstore.similarity_search(query,k=2)
    context = '{ "Context" : "' + "\n".join([x.page_content for x in docs]) + '"}\n\n'
    user_query = f"Now answer this query from the given context : \n\nQuery : {query}"
    prompt = '''You are Air India Customer Service Bot.\nYou have to answer customer queries from the given context.\n\n''' + context + user_query
    print(context)
    with st.chat_message("assistant"):
        st.write(llm(prompt))

def getintent(query,llm):

    sys_prompt = '''You are a AIR INDIA customer service bot.

Your job is to select tools to perform customer tasks.

Here are the list of tools available:

1. ENQUIRY : use this tool to answer flight enquiries

2. FAQ : use this tool to answer customer's doubts or questions other than flight enquiries

3. BOOK : use this tool to book flight tickets

Choose any one of the given tools to complete customer tasks.

Give your response in the following json format:

{
    "TOOL" : Name of the tool should be one of ["ENQUIRY", "FAQ", "BOOK"]
}

'''

    user_prompt = f"Customer : {query}"

    prompt = sys_prompt + user_prompt

    return llm(prompt)

query = st.chat_input("Enter your query here")

if query:

    intent = json.loads(getintent(query,llm).strip())
    print(intent)

    if intent['TOOL'] == 'BOOK':

        with st.chat_message("user"):
            st.write(query)

        with st.form("booking"):

            origin = st.text_input(label="Origin",key="origin")
            destination = st.text_input(label="Destination",key='destination')
            st.form_submit_button("Submit",on_click=setbooking)

    elif intent['TOOL'] == 'ENQUIRY':

        with st.chat_message("user"):
            st.write(query)

        enquiry(query,llm,extractor,fn)

    elif intent['TOOL'] == 'FAQ':
        with st.chat_message("user"):
            st.write(query)
        faq(query,docstore,llm)
