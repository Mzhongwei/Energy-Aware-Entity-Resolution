import multiprocessing as mp

import gensim.models as models
import numpy as np
from gensim.models import Doc2Vec, FastText, Word2Vec


def initialize_embeddings(
    dimensions,
    window_size,
    negative,
    epochs,
    min_count,
    training_algorithm="word2vec",
    learning_method="skipgram",
    workers=mp.cpu_count(),
    sampling_factor=0.001,
):
    """Function used to train the embeddings based on the given walks corpus. Multiple parameters are available to
    tweak the training procedure. The resulting embedding file will be saved in the given path to be used later in the
    experimental phase.

    :param output_embeddings_file: path to save the embeddings file into.
    :param dimensions: number of dimensions to be used when training the model
    :param window_size: size of the context window
    :param training_algorithm: either fasttext or word2vec.
    :param learning_method: skipgram or CBOW
    :param workers: number of CPU workers to be used in during the training. Default = mp.cpu_count().
    """
    if training_algorithm == "word2vec":
        if learning_method == "skipgram":
            sg = 1
        elif learning_method == "CBOW":
            sg = 0
        else:
            raise ValueError("Unknown learning method {}".format(learning_method))
        
        model = Word2Vec(
            vector_size=dimensions,
            window=window_size,
            min_count=min_count,
            sg=sg,
            workers=workers,
            sample=sampling_factor,
            negative=negative,
            epochs=epochs
        )
        return model
    
    elif training_algorithm == "doc2vec":
        if learning_method == "skipgram":
            sg = 1
        elif learning_method == "CBOW":
            sg = 0
        else:
            raise ValueError("Unknown learning method {}".format(learning_method))

        model = Doc2Vec(
            size=dimensions,
            window=window_size,
            min_count=min_count,
            sg=sg,
            workers=workers,
            sample=sampling_factor,
            negative=negative,
            epochs=epochs
        )
        return model

    elif training_algorithm == "fasttext":
        print("Using Fasttext")
        
        model = FastText(
            vector_size=dimensions,
            workers=workers,
            min_count=min_count,
            window=window_size,
            negative=negative,
            epochs=epochs
        )
        return model
