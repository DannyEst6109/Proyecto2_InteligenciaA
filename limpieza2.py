import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import string
import re

# Descarga los recursos necesarios de nltk (stop words, tokenizador, lematizador)
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')

# Carga el archivo CSV
data = pd.read_csv('spam-2.csv', encoding='latin1')




# Expansión de abreviaturas comunes
abbreviations = {
    "lol": "laugh out loud",
    "u": "you",
    "ur": "your",
    "r": "are",
    "gr8": "great",
    "2day": "today",
    "4u": "for you"
}

# Inicializar el lematizador y el derivador (stemmer)
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Define una función para limpiar los mensajes
def clean_text(text, lemmatize=False, expand_abbreviations=True, remove_emoticons=True):
    # Expande abreviaturas comunes
    if expand_abbreviations:
        for key, value in abbreviations.items():
            text = text.replace(key, value)
    # Elimina emoticones
    if remove_emoticons:
        text = re.sub(r'[:;=8][\'-]?[)D]', '', text)
    # Tokeniza el texto en palabras individuales
    tokens = word_tokenize(text)
    # Convierte las palabras a minúsculas y elimina palabras repetidas
    tokens = [word.lower() for word in tokens]
    tokens = [re.sub(r'(.)\1+', r'\1\1', word) for word in tokens]  # Reduce letras repetidas
    # Elimina la puntuación
    table = str.maketrans('', '', string.punctuation)
    stripped = [word.translate(table) for word in tokens]
    # Filtra las stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in stripped if word not in stop_words]
    # Aplica lematización si se especifica
    if lemmatize:
        words = [lemmatizer.lemmatize(word) for word in words]
    # Une las palabras nuevamente en un string
    clean_text = ' '.join(words)
    return clean_text

# Aplica la limpieza de texto a la columna v2 (mensajes) con lematización, expansión de abreviaturas y eliminación de emoticones
data['v2'] = data['v2'].astype(str).apply(lambda x: clean_text(x, lemmatize=True, expand_abbreviations=True, remove_emoticons=True))

# Guarda los datos limpios en un archivo CSV distinto sobrescribiendo la columna original 'v2'
data.to_csv('clean_spam_data.csv', index=False)

# Muestra los primeros mensajes limpios con lematización, expansión de abreviaturas y eliminación de emoticones
print("Mensajes limpios con lematización, expansión de abreviaturas y eliminación de emoticones:")
print(data['v2'].head())
