import pickle
import os


def create_pickle(value=None, filename=None):
    """
    Serializes and saves a Python object to a file using pickle.

    This function takes a Python object and a filename, and serializes the object to a file with the specified name.
    If either the value or the filename is not provided, the function raises an exception.

    Parameters:
    - value: The Python object to serialize. Must not be None for the operation to proceed.
    - filename: The name of the file where the serialized object should be saved. Must not be None for the operation to proceed.

    Raises:
    - Exception: If either `value` or `filename` is None, indicating incomplete arguments for the operation.
    """
    if value is not None and filename is not None:
        with open(filename, "wb") as file:
            pickle.dump(value, file)
    else:
        raise Exception("Pickle file is empty".capitalize())
