import pickle

classifier = pickle.load(open("classifierSC.pkl", "rb"))
vectorizer = pickle.load(open("vectorizerSC.pkl", "rb"))

while True:
    text = input("Type a sentence: ")
    prediction = classifier.predict(vectorizer.transform([text]))

    if prediction[0] == 0:
        print("Female")
    else:
        print("Male")