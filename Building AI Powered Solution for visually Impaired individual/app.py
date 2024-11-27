import streamlit as st
import base64
from gtts import gTTS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pytesseract  # For OCR
from PIL import Image  # For image processing
#import re  # For cleaning extracted text

# Ensure Tesseract-OCR is configured correctly
# Update the tesseract_cmd path based on your installation
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Configuration
GOOGLE_API_KEY = ""  # Replace with Google API key
MODEL_NAME = "gemini-1.5-flash"         # Replace with the appropriate model name

# Function to encode uploaded image to Base64
def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode("utf-8")
    
# Function to create LangChain chain
def create_langchain_chain(system_prompt, human_prompt):
    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
    output_parser = StrOutputParser()
    chat_model = ChatGoogleGenerativeAI(
        google_api_key=GOOGLE_API_KEY, model=MODEL_NAME
    )
    return chat_prompt | chat_model | output_parser

def extract_text_from_image(image_path):
    """
    Extract text from an image using Tesseract OCR.

    Parameters:
    image_path (str): Path to the input image.

    Returns:
    str: Extracted and cleaned text from the image.
    """
    try:
        # Open the image using PIL
        image = Image.open(image_path)
        # Perform OCR using pytesseract
        extracted_text = pytesseract.image_to_string(image)
        #print(f"Raw Extracted Text:\n{extracted_text}")

        # Clean the extracted text to remove unnecessary special characters
        # Remove special characters and extra spaces while keeping punctuation and letters
        #cleaned_text = re.sub(r"[^a-zA-Z0-9.,:!?'\s]", "", extracted_text)
        # Remove extra whitespace
        #cleaned_text = " ".join(cleaned_text.split())
        #print(f"Cleaned Text:\n{cleaned_text}")
        return extracted_text
    except Exception as e:
        print(f"Error during text extraction: {e}")
        return ""

# Function to generate audio from text
def generate_audio(text, filename="output_audio.wav"):
    tts = gTTS(text)
    tts.save(filename)
    return filename


# Main Streamlit Application
def main():

    # Set the page title (this will reflect on the browser tab)
    st.set_page_config(page_title="Vision AI Assistant", layout="centered")

    
    # Application Title
    st.title("""AI Assistant for Visually Impaired 👨‍🦯‍➡️...🕯️""")

    # App Sidebar 
    st.sidebar.markdown("""
        ## ⚡Embark on a journey through👨‍🦯‍➡️...🕯️⚡\n\n
    **🌍 Real-Time Scene Understanding**:\n
    It describes the surroundings to help users understand what's around them.

    **🔊 Text-to-Speech Conversion**:\n
    It reads out text, labels, and signs, so users can hear important information.

    **🛑 Object and Obstacle Detection**:\n 
    It finds objects and obstacles to ensure safe movement.

    **🤖 Personalized Help for Daily Tasks**:\n 
    It helps with everyday tasks like recognizing items and reading labels.
"""
    )
    # Image Uploader
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    # Functionality Selection
    functionality = st.selectbox(
        "Select the functionality you want to use:",
        ["Describe Image", "Text-to-Speech Conversion", "Object Detection", "Task Assistance"],
        index=0,
    )

    # Start Analysis Button
    btn_click = st.button("Start Analysis")

    if uploaded_file and btn_click:

        # Encode image as Base64
        image_data = encode_image(uploaded_file)

        # Generate description or extract text based on functionality
        with st.spinner("👨‍🦯‍➡️ Analyzing the image... Please wait."):
            # AI Prompt Configuration
            if functionality == "Describe Image":
                system_prompt = (
                    "system",
                    """You are a helpful AI Assistant for visiually Impaired Individuals.
                    visually Impaired individuals often face challenges in understanding their environment, reading visual content, and performing tasks that rely on sight.
                    visually Impaired individuals will upload a image and ask you questions related to real-time scene understanding, text-to-speech conversion for reading visual content, object and obstacle detection for safe navigation and personalized assistance for daily tasks
                    you are expected to answer every question and provide comprehensive explanation easy to understandable.
                    """
                )
                input_prompt = "Provide me a detailed description of the uploaded image"

                human_prompt = (
                "human",
                [
                    {"type": "text", "text": "{input}"},
                    {"type": "image_url", "image_url": "data:image/jpeg;base64,{image_data}"},
                ],
                )

                chain = create_langchain_chain(system_prompt, human_prompt)

                user_input = {
                    "input": input_prompt,
                    "image_data": image_data,
                }

                description = chain.invoke(user_input)
                st.image(uploaded_file, caption="Uploaded Image", width=150)
                # Display the description
                st.markdown("### 📝 Analysis Result")
                st.success(description)
            elif functionality == "Text-to-Speech Conversion":
                text_description = extract_text_from_image(uploaded_file)
                st.image(uploaded_file, caption="Uploaded Image", width=150)
                # Display the description
                st.markdown("### 📝 Analysis Result")
                st.success(text_description)
                #text_to_speech(text_description)
                audio_file = generate_audio(text_description)
                # Display Generated Audio
                st.markdown("### 🎧 Generated Audio")
                st.audio(audio_file, format="audio/mp3")
            elif functionality == "Object Detection":
                system_prompt = (
                    "system",
                    """You are a helpful AI Assistant for visiually Impaired Individuals.
                    visually Impaired individuals often face challenges in understanding their environment, reading visual content, and performing tasks that rely on sight.
                    visually Impaired individuals will upload a image and ask you questions related to real-time scene understanding, text-to-speech conversion for reading visual content, object and obstacle detection for safe navigation and personalized assistance for daily tasks
                    you are expected to answer every question and provide comprehensive explanation easy to understandable.
                    """
                )
                input_prompt = """Analyze the provided image to identify all objects or obstacles within it. 
                Highlight the objects with detailed descriptions, including their positions and relevance to the scene. 
                Based on the identified objects, offer actionable insights to enhance user safety and situational awareness. 
                """
                human_prompt = (
                "human",
                [
                    {"type": "text", "text": "{input}"},
                    {"type": "image_url", "image_url": "data:image/jpeg;base64,{image_data}"},
                ],
                )

                chain = create_langchain_chain(system_prompt, human_prompt)

                user_input = {
                    "input": input_prompt,
                    "image_data": image_data,
                }

                description = chain.invoke(user_input)
                st.image(uploaded_file, caption="Uploaded Image", width=150)
                # Display the description
                st.markdown("### 📝 Analysis Result")
                st.success(description)
            elif functionality == "Task Assistance":
                system_prompt = (
                    "system",
                    """You are a helpful AI Assistant for visiually Impaired Individuals.
                    visually Impaired individuals often face challenges in understanding their environment, reading visual content, and performing tasks that rely on sight.
                    visually Impaired individuals will upload a image and ask you questions related to real-time scene understanding, text-to-speech conversion for reading visual content, object and obstacle detection for safe navigation and personalized assistance for daily tasks
                    you are expected to answer every question and provide comprehensive explanation easy to understandable.
                    """
                )
                input_prompt = """Examine the uploaded image and provide tailored guidance to assist with specific daily tasks.
                Recognize and identify items in the image, interpret labels, and offer detailed context-specific information.
                """
                # Focus on actionable insights that help the user efficiently complete their tasks.
                # Ensure the response is clear, practical, and easy to understand, aligning with the user's needs for daily assistance.
                human_prompt = (
                "human",
                [
                    {"type": "text", "text": "{input}"},
                    {"type": "image_url", "image_url": "data:image/jpeg;base64,{image_data}"},
                ],
                )

                chain = create_langchain_chain(system_prompt, human_prompt)

                user_input = {
                    "input": input_prompt,
                    "image_data": image_data,
                }

                description = chain.invoke(user_input)
                st.image(uploaded_file, caption="Uploaded Image", width=150)
                # Display the description
                st.markdown("### 📝 Analysis Result")
                st.success(description)

if __name__=="__main__":
    main()