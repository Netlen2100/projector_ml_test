import pandas as pd
import nltk
import spacy
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error as mse



#Read data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

NLP = spacy.load('en_core_web_sm')
lemma = nltk.stem.wordnet.WordNetLemmatizer()

def text_process(text):
    lm = []
    text_nlp = NLP(text)
    for t in text_nlp:
        lm.append(t.lemma_)
        
    text_nlp = " ".join(lm)
    text_nlp = text_nlp.translate(str.maketrans("", "", string.punctuation))
    text_nlp = [word for word in text_nlp.split() if word.lower() not in nltk.corpus.stopwords.words('english')]
    return " ".join(text_nlp)


train_data['processed_text'] = train_data['excerpt'].apply(text_process)
test_data['processed_text'] = test_data['excerpt'].apply(text_process)

train_data['processed1'] = train_data.processed_text.str.replace(r"[0-9]","")
test_data['processed1'] = test_data.processed_text.str.replace(r"[0-9]","")

X_train, X_test, y_train, y_test = train_test_split(train_data['processed1'] , train_data['target'] )

vectorizer = TfidfVectorizer()
processed  = vectorizer.fit(train_data['processed1'] ) 
processed_test  = vectorizer.transform(test_data['processed1'] ) 
X_train = vectorizer.transform(X_train) 
X_test = vectorizer.transform(X_test )

ridge = Ridge(fit_intercept = True)

X = X_train
y = y_train

ridge.fit(X, y)

y_pred_ridge = ridge.predict(X_test)

MSE = mse(y_test, y_pred_ridge)

print(MSE)
