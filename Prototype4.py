import streamlit as st
import os
import requests
from openai import OpenAI
from bs4 import BeautifulSoup
import pandas as pd
from opencage.geocoder import OpenCageGeocode
from io import BytesIO
import firebase_admin
from firebase_admin import credentials, firestore
import folium
from streamlit_folium import st_folium
from PIL import Image
from PyPDF2 import PdfReader

# Initialize Firebase app
cred = credentials.Certificate("E:/VSC-Python/floodguard-ai-firebase-adminsdk-1gehw-297a26cec3.json")  # Update with your Firebase credentials path
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

# Initialize Firestore client
db = firestore.client()

# Initialize OpenCage Geocoder
geocoder = OpenCageGeocode("469906be508849a68838fbcb10c31ce0")  # Replace with your OpenCage API key

# Initialize OpenAI API client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Santa Clara County zip codes
santa_clara_zip_codes = {
    "95002", "95008", "95013", "95014", "95020", "95032", "95035", "95037",
    "95046", "95101", "95110", "95111", "95112", "95113", "95116", "95117",
    "95118", "95119", "95120", "95121", "95122", "95123", "95124", "95125",
    "95126", "95127", "95128", "95129", "95130", "95131", "95132", "95133",
    "95134", "95135", "95136", "95138", "95139", "95140", "95141", "95148",
    "95150", "95151", "95152", "95153", "95154", "95155", "95156", "95157",
    "95158", "95159", "95160", "95161", "95164", "95170", "95172", "95173",
    "95190", "95191", "95192", "95193", "95194", "95196"
}

# Streamlit page configuration
st.set_page_config(page_title="Flood Preparedness & Reporting", layout="wide")

# Main navigation
st.sidebar.title("Navigation")
option = st.sidebar.selectbox("Choose a tab", ["Main Page", "Flood Information Extractor", "Flood Preparedness Advisor", "Community Flood Reporting Map"])

# Wrapper function for OpenAI API completion
def get_completion(prompt, model="gpt-3.5-turbo"):
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": "You are a flood preparedness expert."},
                      {"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error generating advice: {str(e)}"

# Function to extract flood-related information from a URL
def extract_flood_info_from_url(url, keyword=None, max_paragraphs=5):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.title.string if soup.title else 'No title found'
        paragraphs = [para.get_text().strip() for para in soup.find_all('p') if para.get_text().strip()]
        
        if keyword:
            paragraphs = [para for para in paragraphs if keyword.lower() in para.lower()]
        
        content_text = " ".join(paragraphs[:max_paragraphs])
        summary = summarize_text(content_text)  # Generate summary of the content

        return title, paragraphs[:max_paragraphs], summary
    except Exception as e:
        return str(e), [], None

# Function to generate a summary of the flood-related information
def summarize_text(text, max_tokens=100):
    try:
        prompt = f"Summarize the following flood-related information:\n\n{text}"
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a summarizer for flood-related content."},
                      {"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error generating summary: {str(e)}"
    
# Add a function to handle user questions about the page content
def answer_question_about_content(content, question):
    try:
        prompt = f"Based on the following flood-related information, answer the question:\n\nContent: {content}\n\nQuestion: {question}"
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a flood preparedness expert answering questions."},
                      {"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error answering question: {str(e)}"
    
# Function to fetch flood reports from Firebase Firestore
def fetch_flood_reports():
    docs = db.collection("flood_reports").stream()
    return [{"lat": doc.get("lat"), "lon": doc.get("lon"), "type": doc.get("type"), "severity": doc.get("severity"), "address": doc.get("address"), "image": doc.get("image")} for doc in docs]

# Function to get latitude and longitude from an address
def get_lat_lon(address):
    result = geocoder.geocode(address)
    if result:
        return result[0]['geometry']['lat'], result[0]['geometry']['lng']
    else:
        st.error("Could not find the location. Please enter a valid address.")
        return None, None

# Function to extract text from PDF    
def extract_text_from_pdf(pdf_path):
    try:
        pdf_reader = PdfReader(pdf_path)
        text = ""
        for page in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page].extract_text()
        return text
    except Exception as e:
        return f"Error reading the PDF: {str(e)}"

# Function to get preparedness advice from PDF content
def get_preparedness_advice_from_pdf(pdf_content, zip_code, residence_type, has_pets, wheelchair_accessibility, health_risks):
    try:
        prompt = (
            f"Using the following flood preparedness PDF content, provide advice:\n\n{pdf_content}\n\n"
            f"Considerations: Zip code {zip_code}, residence type {residence_type}, "
            f"pets: {'Yes' if has_pets else 'No'}, "
            f"wheelchair accessibility: {'Yes' if wheelchair_accessibility else 'No'}, "
            f"health risks: {health_risks}."
        )
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a flood preparedness advisor using PDF-based information."},
                      {"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error generating advice: {str(e)}"
    
# Function to analyze the flood image and identify the type of flood
def analyze_flood_image(image_bytes):
    try:
        # OpenAI API call to analyze the image (assuming using OpenAI's GPT-4 Vision or similar model)
        response = openai.Image.create(
            file=image_bytes,
            model="gpt-4-vision-preview"  # Use the correct model if different
        )
        # Assuming the API returns a description of the image
        description = response['choices'][0]['text']
        return description
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

# Handle different options
if option == "Main Page":
    st.title("Flood Preparedness & Reporting System")
    st.write("This tool provides resources to stay safe during floods and report flood incidents in your area.")

elif option == "Flood Information Extractor":
    st.subheader("Flood Information Extractor")

    # Initialize session state variables if they don’t exist
    if 'url_input' not in st.session_state:
        st.session_state.url_input = ''
    if 'keyword_input' not in st.session_state:
        st.session_state.keyword_input = ''
    if 'summary' not in st.session_state:
        st.session_state.summary = ''
    if 'key_points' not in st.session_state:
        st.session_state.key_points = []
    if 'question_input' not in st.session_state:
        st.session_state.question_input = ''
    if 'answer' not in st.session_state:
        st.session_state.answer = ''

    # User inputs for URL, keyword, and maximum paragraphs to display
    st.session_state.url_input = st.text_input("Enter the URL of the flood-related website:", st.session_state.url_input)
    st.session_state.keyword_input = st.text_input("Optional: Specify a flood-related term:", st.session_state.keyword_input)
    max_paragraphs = st.slider("Number of key points to display:", 1, 20, 5)

    # Button to extract flood information
    if st.button("Extract Flood Info"):
        if st.session_state.url_input:
            # Extract title, key points, and summary from the URL
            title, st.session_state.key_points, st.session_state.summary = extract_flood_info_from_url(
                st.session_state.url_input, 
                keyword=st.session_state.keyword_input, 
                max_paragraphs=max_paragraphs
            )
            
            # Display title and summary
            st.write(f"**Page Title:** {title}")
            st.write("### Summary of Flood Information:")
            st.write(st.session_state.summary if st.session_state.summary else "No summary available.")
            
            # Display key flood information as bullet points
            st.write("### Key Flood Information:")
            for i, point in enumerate(st.session_state.key_points, 1):
                st.write(f"{i}. {point}")

    # Input for user question
    st.session_state.question_input = st.text_input("Ask a specific question about this page's content:", st.session_state.question_input)
    
    # Button to generate an answer based on the question input
    if st.button("Get Answer") and st.session_state.question_input:
        st.session_state.answer = answer_question_about_content(
            f"{st.session_state.summary} {' '.join(st.session_state.key_points)}", 
            st.session_state.question_input
        )
        st.write("### Answer:")
        st.write(st.session_state.answer)
        
# In the Flood Preparedness Advisor section of your Streamlit app:
elif option == "Flood Preparedness Advisor":
    if option == "Flood Preparedness Advisor":
        st.subheader("Flood Preparedness Advisor")
    
    # Path to the PDF file
    pdf_path = r"E:\VSC-Python\Get Flood Ready Essential Tips _ Santa Clara Valley Water.pdf"
    
    # Extract text from the PDF
    pdf_content = extract_text_from_pdf(pdf_path)

    # Form for user inputs
    with st.form(key="advisor_form"):
        zip_code = st.text_input("Enter your zip code (Santa Clara County only)")
        residence_type = st.selectbox("Type of residence", ["House", "Apartment", "Mobile Home", "Other"])
        has_pets = st.checkbox("Do you have pets?")
        wheelchair_accessibility = st.checkbox("Wheelchair accessibility considerations")
        
        # Add field for health risks
        health_risks = st.text_area("List any health risks you might have during flooding", 
                                    help="E.g., respiratory issues, allergies, or other health conditions.")
        
        submitted = st.form_submit_button("Get Preparedness Advice")
        
        if submitted and zip_code in santa_clara_zip_codes:
            # Get the preparedness advice from the PDF content and user inputs
            response = get_preparedness_advice_from_pdf(
                pdf_content, zip_code, residence_type, has_pets, wheelchair_accessibility, health_risks
            )
            st.write(response)
        else:
            st.warning("Please check the path of the PDF file to ensure it's correct.")
elif option == "Community Flood Reporting Map":
    st.subheader("Community Flood Reporting Map")
    if 'flood_data' not in st.session_state:
        st.session_state.flood_data = fetch_flood_reports()
    
    # Sidebar form for reporting a flood incident
    with st.sidebar.form("flood_form"):
        street_address = st.text_input("Street Address")
        flood_type = st.selectbox("Cause of Flood", ["Storm Drain Blockage", "Well/Reservoir Overflow", "Pipe Burst", "Debris", "Other"])
        custom_flood_type = st.text_input("Specify cause of flooding") if flood_type == "Other" else flood_type
        severity = st.slider("Flood Severity (1 = Minor, 5 = Severe)", 1, 5)
        image = st.file_uploader("Upload a flood image", type=["jpg", "png", "jpeg"])
        submitted = st.form_submit_button("Submit Report")

        if submitted and street_address:
            lat, lon = get_lat_lon(street_address)
            if lat and lon:
                flood_entry = {"lat": lat, "lon": lon, "type": custom_flood_type, "severity": severity, "address": street_address, "image": image.read() if image else None}
                st.session_state.flood_data.append(flood_entry)
                db.collection("flood_reports").add(flood_entry)
                st.success(f"Flood report added at {street_address}.")

    # Create and display map with flood data
    df = pd.DataFrame(st.session_state.flood_data)
    if not df.empty:
        m = folium.Map(location=[df["lat"].mean(), df["lon"].mean()], zoom_start=12)
        for _, row in df.iterrows():
            folium.Marker(location=[row["lat"], row["lon"]], popup=f"Address: {row['address']}<br>Type: {row['type']}<br>Severity: {row['severity']}", icon=folium.Icon(color="red")).add_to(m)
        st_folium(m, width=700, height=500)
        st.write("### Reported Flood Incidents:")
        for idx, row in df.iterrows():
            st.write(f"**Location**: {row['address']}, **Type**: {row['type']}, **Severity**: {row['severity']}")
            if row['image']:
                image = Image.open(BytesIO(row['image']))
                st.image(image, caption=f"Flood at {row['address']}", use_column_width=True)
