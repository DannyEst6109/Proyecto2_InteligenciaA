import pandas as pd
from sklearn.model_selection import train_test_split

# Carga de datos
data = pd.read_csv("clean_spam_data.csv")

# Agregar la longitud del mensaje como una característica adicional
data['message_length'] = data['v2'].apply(lambda x: len(str(x)))

# Agregar la frecuencia de palabras clave como características adicionales
keywords = ["free", "win", "prize", "claim", "urgent"]
for keyword in keywords:
    data[keyword + '_count'] = data['v2'].apply(lambda x: str(x).lower().count(keyword))

# Definir características y variable objetivo
X = data[['v2', 'message_length'] + [keyword + '_count' for keyword in keywords]]
y = data['v1']

# Entrenamiento y conjunto de prueba (80% para entrenamiento, 20% para prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Calculando probabilidades para el modelo bayesiano
N_s_train = y_train.value_counts()['spam']
N_h_train = y_train.value_counts()['ham']

P_S_train = N_s_train / len(y_train)
P_H_train = N_h_train / len(y_train)

# Nw,s y Nw,h
word_counts_spam_train = {} 
word_counts_ham_train = {} 

for text, label in zip(X_train['v2'], y_train):
    words = str(text).split()
    for word in words:
        if label == 'spam':
            word_counts_spam_train[word] = word_counts_spam_train.get(word, 0) + 1
        else:
            word_counts_ham_train[word] = word_counts_ham_train.get(word, 0) + 1

# P(W|S) y P(W|H)
P_W_given_S_train = {}
P_W_given_H_train = {}

for word in word_counts_spam_train:
    P_W_given_S_train[word] = word_counts_spam_train[word] / N_s_train

for word in word_counts_ham_train:
    P_W_given_H_train[word] = word_counts_ham_train[word] / N_h_train

# P(W) para cada palabra en los datos de entrenamiento
P_W_train = {}
for word in word_counts_spam_train:
    if word in word_counts_ham_train:
        P_W_train[word] = (word_counts_spam_train[word] / N_s_train) / ((word_counts_ham_train[word] / N_h_train) + (word_counts_spam_train[word] / N_s_train))

# Funciones para calcular las probabilidades de que un texto sea spam dado ciertas palabras
def calculate_P_S_given_W_train(word):
    P_W_given_S = P_W_given_S_train.get(word, 0)
    P_W_given_H = P_W_given_H_train.get(word, 0)
    
    if P_W_given_S == 0 and P_W_given_H == 0:
        return 0
        
    numerator = P_W_given_S * P_S_train
    denominator = numerator + (P_W_given_H * P_H_train)
    
    if denominator == 0:
        return 0  
    
    return numerator / denominator

def calculate_P_S_given_W_text(text):
    probabilities = []
    words = str(text).split()

    for word in words:
        Prob_W = calculate_P_S_given_W_train(word)
        probabilities.append(Prob_W)
    
    probabilities = [prob for prob in probabilities if prob != 0]
    
    if probabilities == []:
        return 0
    
    numerator = probabilities[0]
    denominator = 1 - probabilities[0]
    
    for prob in probabilities[1:]:
        numerator *= prob
        denominator *= (1 - prob)
    
    if denominator == 0 and numerator == 0:
        return 0

    return numerator / (numerator + denominator)

def predict_spam_or_ham(text):
    prob_spam = calculate_P_S_given_W_text(text)
    words_found = []
    words = str(text).split()
    for word in words:
        if word in P_W_train:
            words_found.append(word)
    return prob_spam, 'spam' if prob_spam > 0.5 else 'ham', words_found

# Ejemplo de uso
#input_text = "CALL to No: 86888 & claim your reward of 3 hours talk time to use from your phone now!"
#input_text = "URGENT! Your mobile won a 2,000 bunus caller prize on 02/06/03! this is the 2nd attempt to reach YOU!"
#input_text = "I Will come tomorrow di"
#input_text = "Now, whats your house # again ? And do you have any beer there ?"
input_text = "Reminder. You have not downloaded the content you have already paid for."

prob_spam, prediction, words_found = predict_spam_or_ham(input_text)
print("Prediction:", prediction)
print("probabilidad de ser spam:", prob_spam)
print("palabras encontradas:", words_found)
