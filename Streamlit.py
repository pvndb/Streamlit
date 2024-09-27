import streamlit as st
# from streamlit_chat import message
# import bedrock_app as bedrock
# from langchain.callbacks.stdout import StdOutCallbackHandler
import json
import os
import boto3
from Chatbot import get_bedrock_client, retrieve, invoke_claude_3

bedrock_client, bedrock_agent_client = get_bedrock_client(
    assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
    region=os.environ.get("AWS_DEFAULT_REGION", None)
)

kb_dict = {"knowledge-base-Targa": "2J5BA4RDMO"
            }


def clear_input():
    """Clear input when clicking `Clear conversation`."""
    st.session_state["temp"] = st.session_state["input"]
    st.session_state["input"] = ""
    
# Page config
st.set_page_config(page_title="NexEra Trinity", page_icon=":book:")
st.warning('Trinity can make mistakes. Consider double-checking. Test version updated on Sep 26, 2024. ', icon="🌱")

st.markdown(
    """
        <style>
            .appview-container .main .block-container {{
                padding-top: {padding_top}rem;
                padding-bottom: {padding_bottom}rem;
                }}

        </style>""".format(
        padding_top=2.4, padding_bottom=0
    ),
    unsafe_allow_html=True)

# Using the custom classes with Markdown for different parts of the header
st.markdown("""
    <style>
    .part1 {
        font-size:40px !important;
        font-weight: bold;
        color: #F6A81C;
        padding-top: 0rem;
        padding-bottom: 0rem;
    }
    .part2 {
        font-size:35px !important;
        font-weight: bold;
        color: #00954B;
        padding-top: 0rem;
        padding-bottom: 0rem;
    }
    </style>
    """, unsafe_allow_html=True)
st.markdown('<p><span class="part1">TRINITY Xpert </span></p>', unsafe_allow_html=True)
st.markdown('<p style="font-size: 20px; color: #808080; font-style: italic; padding-top: 0rem;">Effortless Gas Contract and Invoice Reconciliation</p>', unsafe_allow_html=True)


kb = st.selectbox(
    "What would you like to inquire?",
    tuple(kb_dict.keys()),
    index=0,
    placeholder="Select your knowledge base...",)
kb_id = kb_dict[kb]

selectbox_font_css = """
    <style>
    div[class*="stSelectbox"] label p {
        font-size: 18px;
        color: #00954B;
        font-style: bold;
    }
    div[class*="stLabel"] label p {
        font-size: 14px;
    }
    </style>
    """
st.write(selectbox_font_css, unsafe_allow_html=True)

if "questions" not in st.session_state:
    st.session_state.questions = []
if "answers" not in st.session_state:
    st.session_state.answers = []
    
with st.form("input-form", border=False): #clear_on_submit=True,
    user_input = st.text_area("Enter a question here:", value="", height=15, key="input")
    click_button = st.form_submit_button("Submit")

form_font_css = """
    <style>
    div[class*="stTextArea"] label p {
        font-size: 18px;
        color: #00954B;
        font-style: bold;
    }
    div[class*="stTextInput"] label p {
        font-size: 14px;
    }
    </style>
    """
st.write(form_font_css, unsafe_allow_html=True)

# Clear conversation
clear = st.button("Clear Conversation", key="clear", on_click=clear_input)
if clear:
    st.session_state.questions = []
    st.session_state.answers = []

if click_button:
    # Catch the error
    if "questions" not in st.session_state:
        st.session_state.questions = []
    _, context = retrieve(bedrock_agent_client=bedrock_agent_client, query=user_input, kb_id=kb_id, numberOfResults=5)
    answer = invoke_claude_3(bedrock_client=bedrock_client, query=user_input, context=context)
    
    with st.expander("Reference"):
        st.markdown("**Source documents were retrieved:**")
        for i in range(len(context)):
            st.markdown("- " + context[i])

    st.session_state.questions.append(user_input)
    st.session_state.answers.append(answer)
    
    if st.session_state["answers"]:
        for i in range(len(st.session_state["answers"]) - 1, -1, -1):
            ai = st.chat_message("assistant")
            ai.write(st.session_state["answers"][i])
            
            human = st.chat_message("user")
            human.write(st.session_state["questions"][i])