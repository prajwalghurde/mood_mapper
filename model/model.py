import pandas as pd
import neattext.functions as nfx

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib

#data set load
df = pd.read_csv("C:\\Users\\ashut\\OneDrive\\Desktop\\hackathondevcraft\\Text-Emotion-Detection\\Text Emotion Detection\\data\\emotion_dataset_raw.csv")

# pre -processing data
df['Clean_Text'] = df['Text'].apply(nfx.remove_userhandles)
df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_stopwords)

# splitting data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(df['Clean_Text'], df['Emotion'], test_size=0.3, random_state=42)

# define pipelines
pipe_lr = Pipeline(steps=[('cv', CountVectorizer()), ('lr', LogisticRegression())])
pipe_svm = Pipeline(steps=[('cv', CountVectorizer()), ('svc', SVC(kernel='rbf', C=10))])
pipe_rf = Pipeline(steps=[('cv', CountVectorizer()), ('rf', RandomForestClassifier(n_estimators=10))])


# training models
pipe_lr.fit(x_train, y_train)
pipe_svm.fit(x_train, y_train)
pipe_rf.fit(x_train, y_train)

# Evaluate the models
lr_score = pipe_lr.score(x_test, y_test)
svm_score = pipe_svm.score(x_test, y_test)
rf_score = pipe_rf.score(x_test, y_test)

print("Logistic Regression Accuracy:", lr_score)
print("Logistic Regression Accuracy percentage:", lr_score * 100)
print("SVM Accuracy:", svm_score)
print("SVM Accuracy percentage:", svm_score * 100)  
print("Random Forest Accuracy:", rf_score)
print("Random Forest Accuracy percentage:", rf_score * 100)


#save the best model (current lr model)
best_model = pipe_svm

joblib.dump(best_model, "text_emotion_svm.pkl")
