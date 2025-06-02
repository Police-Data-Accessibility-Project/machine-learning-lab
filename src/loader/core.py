import joblib

class Loader:



    @staticmethod
    def bag_of_words(path):
        return joblib.load(path)