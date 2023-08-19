import config.Config as Config
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequence

def generate(model, encoder, tokenizer, headline, threshold=0.5):
    seq = tokenizer.texts_to_sequences(headline)
    padded_seq = pad_sequence(seq, maxlen=Config.maxlen, padding='post', truncate='post')
    prediction = model.predict(padded_seq)
    predicted_labels = (prediction > threshold).astype(int)
    predicted_categories = encoder.inverse_transform(predicted_labels)
    predictions = []
    for category in predicted_categories:
        predictions.append(category)
    return predictions