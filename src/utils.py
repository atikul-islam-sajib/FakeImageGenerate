import joblib as pickle


def create_pickle(value=None, filename=None):
    if value is not None and filename is not None:
        pickle.dump(value=value, filename=filename)
    else:
        raise Exception("Pickle file is empty".capitalize())
