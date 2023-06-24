import json
from tqdm import tqdm
import logging
import pickle
from collections import Counter
import re
import fire
from build_vocab import Vocabulary

def process(input_json: str,
            prefix_vocab: Vocabulary,
            suffix_vocab: Vocabulary):
    data = json.load(open(input_json, "r"))["audios"]
    skeleton_list = []
    modifier_list = []
    for item in data:
        for caption in item['captions']:
            tem = caption['caption'].split(' ')
            prefix = []
            suffix = []
            for idx, token in enumerate(tem):
                if token in prefix.word2idx.keys():

    json.dump({ "audios": data }, open(input_json, "w"), indent=4)
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word("<pad>")
    vocab.add_word("<start>")
    vocab.add_word("<end>")
    vocab.add_word("<unk>")

    # Add the words to the vocabulary.
    for word in words :
        vocab.add_word(word)
    return vocab

def process(input_json: str,
            output_file: str,
            threshold: int = 1,
            keep_punctuation: bool = False,
            character_level: bool = False,
            host_address: str = "http://localhost:9000",
            zh: bool = True):
    logger = logging.Logger("Build Vocab")
    logger.setLevel(logging.INFO)
    vocabulary = build_vocab(
        input_json=input_json, threshold=threshold, keep_punctuation=keep_punctuation,
        host_address=host_address, character_level = character_level, zh=zh)
    pickle.dump(vocabulary, open(output_file, "wb"))
    logger.info("Total vocabulary size: {}".format(len(vocabulary)))
    logger.info("Saved vocab to '{}'".format(output_file))


if __name__ == '__main__':
    fire.Fire(process)
