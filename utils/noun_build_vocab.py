import json
from tqdm import tqdm
import logging
import pickle
from collections import Counter
import re
import fire
import stanza
import captioning

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx["<unk>"]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(input_json: str,
                threshold: int,
                kw_threshold: int,
                keep_punctuation: bool,
                host_address: str,
                character_level: bool = False,
                zh: bool = True ):
    """Build vocabulary from csv file with a given threshold to drop all counts < threshold

    Args:
        input_json(string): Preprossessed json file. Structure like this: 
            {
              'audios': [
                {
                  'audio_id': 'xxx',
                  'captions': [
                    { 
                      'caption': 'xxx',
                      'cap_id': 'xxx'
                    }
                  ]
                },
                ...
              ]
            }
        threshold (int): Threshold to drop all words with counts < threshold
        keep_punctuation (bool): Includes or excludes punctuation.

    Returns:
        vocab (Vocab): Object with the processed vocabulary
"""
    data = json.load(open(input_json, "r"))["audios"]
    counter = Counter()
    
    if zh:
        from nltk.parse.corenlp import CoreNLPParser
        from zhon.hanzi import punctuation
        parser = CoreNLPParser(host_address)
        for audio_idx in tqdm(range(len(data)), leave=False, ascii=True):
            for cap_idx in range(len(data[audio_idx]["captions"])):
                caption = data[audio_idx]["captions"][cap_idx]["caption"]
                # Remove all punctuations
                if not keep_punctuation:
                    caption = re.sub("[{}]".format(punctuation), "", caption)
                if character_level:
                    tokens = list(caption)
                else:
                    tokens = list(parser.tokenize(caption))
                data[audio_idx]["captions"][cap_idx]["tokens"] = " ".join(tokens)
                counter.update(tokens)
    else:
        from captioning.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
        captions = {}
        for audio_idx in range(len(data)):
            audio_id = data[audio_idx]["audio_id"]
            captions[audio_id] = []
            for cap_idx in range(len(data[audio_idx]["captions"])):
                caption = data[audio_idx]["captions"][cap_idx]["caption"]
                captions[audio_id].append({
                    "audio_id": audio_id,
                    "id": cap_idx,
                    "caption": caption
                })
        tokenizer = PTBTokenizer()
        captions = tokenizer.tokenize(captions)
        noun_counter = Counter()
        stanza_nlp = stanza.Pipeline(lang='en', processors='tokenize, ner')
        for audio_idx in tqdm(range(len(data)), leave=False, ascii=True):
            audio_id = data[audio_idx]["audio_id"]
            for cap_idx in range(len(data[audio_idx]["captions"])):
                tokens = captions[audio_id][cap_idx]
                data[audio_idx]["captions"][cap_idx]["tokens"] = tokens
                counter.update(tokens.split(" "))
                stanza_tokens = [ent for sent in stanza_nlp(tokens).sentences
                        for ent in sent.ents if  ent.type!='CARDINAL']
                noun_counter.update(stanza_tokens)
        import pdb; pdb.set_trace()
        noun_list = [word for word, cnt in noun_counter.items() if cnt >= 1]
        print("\n num noun:", len(noun_list), "\n")
        import random
        for audio_idx in tqdm(range(len(data)), leave=False, ascii=True):
            audio_id = data[audio_idx]["audio_id"]
            for cap_idx in range(len(data[audio_idx]["captions"])):
                tokens = captions[audio_id][cap_idx].split(" ")
                for word_idx in range(len(tokens)):
                    if tokens[word_idx] in noun_list and random.random() > 0.2:
                        rand_idx = random.randint(0, len(noun_list)-1)
                        tokens[word_idx] = noun_list[rand_idx]
                data[audio_idx]["captions"][cap_idx]["swapent"] = " ".join(tokens)

        json.dump({ "audios": data }, open("data/audiocaps/val/swapent.json", "w"), indent=4)
            

def process(input_json: str,
            output_file: str,
            kw_threshold: int = 50,
            threshold: int = 1,
            keep_punctuation: bool = False,
            character_level: bool = False,
            host_address: str = "http://localhost:9000",
            zh: bool = True):
    logger = logging.Logger("Build Vocab")
    logger.setLevel(logging.INFO)
    build_vocab(
        input_json=input_json, threshold=threshold, kw_threshold=kw_threshold,
        keep_punctuation=keep_punctuation,
        host_address=host_address, character_level = character_level, zh=zh)
    logger.info("Saved vocab to '{}'".format(output_file))


if __name__ == '__main__':
    fire.Fire(process)
