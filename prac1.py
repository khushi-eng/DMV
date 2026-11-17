import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download required NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Sample text document
file_path = r"C:\Users\khush\Downloads\IR_pracs\sample.txt"
with open(file_path, "r", encoding="utf-8") as file:
  text = file.read()

print("Original Text:")
print(text)

# 1. Lowercasing and Tokenization
tokens = word_tokenize(text.lower())
print("\nTokens:")
print(tokens)

# 2. Stop Word & Punctuation Removal
stop_words = set(stopwords.words('english'))
filtered_words = [
    word for word in tokens if word.isalpha() and word not in stop_words
]
print("\nAfter Stop Word Removal:")
print(filtered_words)

# 3. Stemming
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in filtered_words]
print("\nAfter Stemming:")
print(stemmed_words)