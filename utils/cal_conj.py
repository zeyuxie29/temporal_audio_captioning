import json
from tqdm import tqdm
import pickle
from collections import Counter
import fire
import stanza


def process(train: str, val:str, test:str, mode:int = 1):
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    tokenizer = PTBTokenizer()
    conj_counter = Counter()
    stanza_conj = ["CCONJ", "SCONJ"]
    stanza_nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma')
    by_counter = Counter()
    s={}
    s['train']= json.load(open(train, "r"))["audios"]
    s['val'] =  json.load(open(val, "r"))["audios"]        
    s['test'] =  json.load(open(test, "r"))["audios"]        
    conj2senten = {}
    if mode == 1:
        splits = ["train", "val", "test"]
        output = "data/audiocaps/all/all_conj_counter.pkl"
    else:
        splits = ["test"]
        output = "data/audiocaps/all/test_conj_counter.pkl" 
    for split in splits:
        data = s[split]
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
        captions = tokenizer.tokenize(captions)
        for audio_idx in tqdm(range(len(data)), leave=False, ascii=True):
            audio_id = data[audio_idx]["audio_id"]
            for cap_idx in range(len(data[audio_idx]["captions"])):
                tokens = captions[audio_id][cap_idx]
                data[audio_idx]["captions"][cap_idx]["tokens"] = tokens
                #stanza_tokens = [word.lemma for sent in stanza_nlp(tokens).sentences 
                #        for word in sent.words if word.upos in stanza_conj]
                #conj_counter.update(stanza_tokens)
                for sent in stanza_nlp(tokens).sentences:
                    for word in sent.words:
                        if word.upos in stanza_conj:
                            if word.lemma in conj2senten.keys():
                                conj2senten[word.lemma].append(tokens)
                            else:
                                conj2senten[word.lemma] = [tokens]

    import pdb; pdb.set_trace()
    pickle.dump(conj_counter, open(output, "wb"))


if __name__ == '__main__':
    fire.Fire(process)
