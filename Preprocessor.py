# Description: This file contains the code for preprocessing the text data. The preprocessing steps include converting the pdf/docx file to text, removing punctuation, removing numbers, removing stopwords, lemmatization, stemming, etc.

# Importing the required libraries
import PyPDF2
import docx2txt
import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Initialize the lemmatizer and stemmer
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

# Download the stopwords, wordnet and respective things from nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Convert the pdf file to text
def pdf_to_text(pdf_path):
    """
    Parameters: pdf_path: str
    Returns: text: str
    """
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ''
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

# Extract text from the file based on the file type
def extract_text(file_path, file_type):
    """
    Parameters: file_path: str, file_type: str
    Returns: text: str
    """
    if file_type == "pdf":
        return pdf_to_text(file_path)
    elif file_type == "docx":
        return docx2txt.process(file_path)
    else:
        return None
    
# Replace "-" with space
def replace_hyphen(text):
    """
    Parameters: text: str
    Returns: text: str
    """
    text = text.replace('-', ' ')
    text = text.replace('–', ' ')
    return text

# Change the text to lowercase
def text_lowercase(text):
    """
    Parameters: text: str
    Returns: text: str
    """
    return text.lower()

# Remove numbers from the text
def remove_numbers(text):
    """
    Parameters: text: str
    Returns: result: str
    """
    result = re.sub(r'\d+', '', text)
    return result

# Remove punctuation from the text
def remove_punctuation(text):
    """
    Parameters: text: str
    Returns: text: str
    """
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

# Remove newlines from the text
def remove_newlines(text):
    """
    Parameters: text: str
    Returns: text: str
    """
    return text.replace('\n', ' ')

# Remove whitespaces from the text
def remove_whitespace(text):
    """
    Parameters: text: str
    Returns: text: str
    """
    return " ".join(text.split())

# remove stopwords function
def remove_stopwords(text):
    """
    Parameters: text: str
    Returns: text: str
    """
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return " ".join(filtered_text)


# Lemmatize the words
def lemma_words(text):
    """
    Parameters: text: str
    Returns: text: str
    """
    word_tokens = word_tokenize(text)
    lemmas = [lemmatizer.lemmatize(word) for word in word_tokens]
    return ' '.join(lemmas)

# Stem the words
def stem_words(text):
    """
    Parameters: text: str
    Returns: text: str
    """
    word_tokens = word_tokenize(text)
    stems = [ps.stem(word) for word in word_tokens]
    return ' '.join(stems)

# Preprocess the text with lemmatization
def preprocess_text_stem(text):
    """
    Parameters: text: str
    Returns: text: str
    """
    text = replace_hyphen(text)
    text = text_lowercase(text)
    text = remove_numbers(text)
    text = remove_punctuation(text)
    text = remove_newlines(text)
    text = remove_whitespace(text)
    text = remove_stopwords(text)
    text = stem_words(text)
    return text

# Preprocess the text with stemming
def preprocess_text_lemma(text):
    """
    Parameters: text: str
    Returns: text: str
    """
    text = replace_hyphen(text)
    text = text_lowercase(text)
    text = remove_numbers(text)
    text = remove_punctuation(text)
    text = remove_newlines(text)
    text = remove_whitespace(text)
    text = remove_stopwords(text)
    text = lemma_words(text)
    return text

# Preprocess the text with lemmatization and stemming both
def preprocess_text_lemma_stem(text):
    """
    Parameters: text: str
    Returns: text: str
    """
    text = replace_hyphen(text)
    text = text_lowercase(text)
    text = remove_numbers(text)
    text = remove_punctuation(text)
    text = remove_newlines(text)
    text = remove_whitespace(text)
    text = remove_stopwords(text)
    text = lemma_words(text)
    text = stem_words(text)
    return text

# Convert list to string
def list2str(text):
    """
    Parameters: text: list
    Returns: text: str
    """
    return ' '.join(text)




