import streamlit as st
import google.generativeai as genai

f = open("D:\DataScience\ProjectsGenAI\keys\geminikey.txt")
key = f.read()
genai.configure(api_key=key)

sys_prompt = """you are a AI code Reveiwer for Python
                Developer's will ask you to review their python code
                you are expected to analyze the submitted code and identify potential bugs,errors or area of improvement
                If you found any error in submitted code, Make sure to provide the Bug Report first followed by Fixed Code
                otherwise If you not found any error provide a message there is no need of any changes
                In case if a developer ask any thing out of code review
                politely decline and tell them to ask questions code review only"""

model = genai.GenerativeModel(model_name="models/gemini-1.5-flash",
                              system_instruction=sys_prompt)

st.title(":robot_face: AI Code Reviewer")

user_prompt = st.text_area(
    "Enter your Python code here ...",
)

btn_click = st.button("Generate", type='secondary')

if btn_click:
    response = model.generate_content(user_prompt)
    if response:
        st.title("Code Review")
        st.write(response.text)