import pandas as pd
import config.Config as Config
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequence

class DataLoader():

    def __init__(self):
        self.data = pd.read_json("/content/data/News_Category_Dataset_v3.json", lines=True)
        self.data['category'] = data['category'].replace('U.S. NEWS', 'NEWS')
        self.tokenizer = Tokenizer(num_words=Config.vocab_size, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(data['headline'])
        self.data_seq = tokenizer.texts_to_sequences(data['headline'])
        self.data_padded = pad_sequence(data_seq, maxlen=Config.maxlen, padding='post', truncate='post')
        self.encoder = OneHotEncoder()
        self.y = self.encoder.fit_transform(self.data['category'].values.reshape(-1,1)).toarray()
        self.X_train, self.y_train, self.X_val, self.y_val = train_test_split(self.data_padded, self.y, test_size=0.3, random_state=42)
    def load_data(self):
        return self.X_train, self.y_train, self.X_val, self.y_val
    def get_encoder(self):
        return self.encoder
    def get_tokenizer(self):
        return self.tokenizer