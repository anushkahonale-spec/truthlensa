import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# Small starter dataset
# 0 = likely misleading/fake
# 1 = likely authentic

texts = [
    "Government announces new education scholarship for students",
    "Scientists publish research findings in a scientific journal",
    "The university announced its examination schedule",
    "The company released its quarterly financial report",
    "The health department published new public information",
    "Researchers conducted a study and published the results",

    "SHOCKING secret truth they don't want you to know",
    "BREAKING unbelievable news share this immediately",
    "You will never believe what happened next",
    "Doctors are hiding this miracle discovery",
    "This secret trick will change your life instantly",
    "Share this before it gets deleted",
]

labels = [
    1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0
]


# Convert text into numerical features
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

X = vectorizer.fit_transform(texts)


# Train classifier
model = LogisticRegression(max_iter=1000)

model.fit(X, labels)


# Save trained model
joblib.dump(model, "models/text_classifier.pkl")
joblib.dump(vectorizer, "models/text_vectorizer.pkl")


print("TruthLens text model trained successfully.")