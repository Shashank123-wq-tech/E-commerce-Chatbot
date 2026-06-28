from groq import Groq
import os
import streamlit as st

client = Groq(
    api_key=st.secrets("GROQ_API_KEY")
)

def generate_response(prompt):

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role":"user",
                "content":prompt
            }
        ]
    )

    return response.choices[0].message.content