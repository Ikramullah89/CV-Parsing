import streamlit as st
import re
import os
import fitz  # PyMuPDF for PDF handling
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download nltk resources if necessary
nltk.download('stopwords')
nltk.download('wordnet')

# Load environment variables
load_dotenv()

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Check for API key
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    st.error("API key not found. Please check your `.env` file or environment settings.")
else:
    st.title('Essentia Technologies Job Description Generator')
    
    # Initialize session_state if not already
    if 'job_description' not in st.session_state:
        st.session_state['job_description'] = ""
    
    # Input fields for job details
    job_title = st.text_input('Enter Job Title:')
    education = st.text_input('Enter Required Education:')
    experience = st.text_input('Enter Required Experience:')

    # Generate Job Description
    if st.button('Generate Description') and job_title and education and experience:
        prompt = (f"You are an expert recruiter at Essentia Technologies. Create a job description for {job_title}, "
                  f"requiring {education} and {experience}. Emphasize company values and learning opportunities.")
        # Simulated job description for example (replace with actual model call if configured)
        st.session_state['job_description'] = f"Sample JD for {job_title}"
        st.subheader('Generated Job Description')
        st.write(st.session_state['job_description'])

    # Allow CV upload if job description exists
    if st.session_state['job_description']:
        st.write("Please upload your CV:")
        uploaded_file = st.file_uploader("Upload your CV", type=["pdf"])

        if uploaded_file:
            # Process PDF file
            pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            pages_text = ''.join(page.get_text() for page in pdf_document)
            pdf_document.close()

            # Text preprocessing function
            def preprocess_text(text):
                text = text.lower()
                text = re.sub(r'[^a-z\s]', '', text)
                stop_words = set(stopwords.words('english'))
                return ' '.join(lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words)

            # Preprocess texts
            job_description_processed = preprocess_text(st.session_state['job_description'])
            pages_text_processed = preprocess_text(pages_text)

            # Calculate similarity
            vectorizer = TfidfVectorizer(max_features=1500, ngram_range=(1, 3), min_df=1)
            vectors = vectorizer.fit_transform([job_description_processed, pages_text_processed])
            similarity_score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

            st.subheader('Cosine Similarity Score')
            st.write(f"Similarity Score: {similarity_score:.2f}")
            if similarity_score > 0.05:
                st.write("Your CV matches well with the job description.")
            else:
                st.write("Your CV does not match the job description closely enough.")
