import json
from tqdm import tqdm
import logging
import pickle
from collections import Counter
import re
import fire
import stanza
import captioning
import numpy as np


def span_calculate(input_json: str, output_file: str):
    data = json.load(open(input_json, "r"))["audios"]
    counter = Counter()
    
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    captions = {}
    if "tokens" not in data[0]["captions"][0].keys():
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
        flag = 1
    else:
        flag = 0
    after_span = ["follow", "followed", "then", "after"]
    while_span = ["while", "and", "when", "with", "as"]
    label_caps_cal = [0, 0, 0, 0]
    label_audios_cal = [0, 0, 0, 0]
    label_audios_dict = {}

    for audio_idx in tqdm(range(len(data)), leave=False, ascii=True):
        audio_id = data[audio_idx]["audio_id"]
        audio_label = 0
        for cap_idx in range(len(data[audio_idx]["captions"])):
            if flag:
                tokens = captions[audio_id][cap_idx]
                data[audio_idx]["captions"][cap_idx]["tokens"] = tokens
            else:
                tokens = data[audio_idx]["captions"][cap_idx]["tokens"]
            token_list = tokens.split(" ")
            counter.update(token_list)
            after_flag, while_flag = 0, 0
            for i in range(len(token_list)):
                if token_list[i] in after_span:
                    after_flag = 2
                if token_list[i] in while_span:
                    while_flag = 1
                if after_flag + while_flag == 3:
                    break
            cap_label = after_flag + while_flag
            if cap_label > audio_label:
                audio_label = cap_label
            data[audio_idx]["captions"][cap_idx]["time_label"] = cap_label
            label_caps_cal[cap_label] += 1
            #if cap_label < audio_label:
                #print(data[audio_idx]["captions"][cap_idx])
        if (audio_id not in label_audios_dict.keys()) or (audio_label > label_audios_dict[audio_id]):
            label_audios_dict[audio_id] = audio_label

    for k, v in label_audios_dict.items():
        label_audios_cal[v] += 1

    #print("caps_cal:", label_caps_cal)
    #print("audios_cal", label_audios_cal)
    #json.dump({ "audios": data }, open(output_file, "w"), indent=4)
    return label_caps_cal, label_audios_cal
    



def process(input_json: str,
        output_file: str = None):
        logger = logging.Logger("Build Vocab")
        logger.setLevel(logging.INFO)
        if output_file == None:
            output_file = "/".join(input_json.split('/')[:-1]) + "/timelabel_text.json"
        tem_caps, tem_audios = span_calculate(input_json, output_file)
        #print(tem_caps, tem_audios)
        caps = np.array(tem_caps)
        #print((caps[2]+caps[3])/caps.sum()*100)
        return caps

def process_all():
    caps = np.zeros(4)
    #for s in ['data/audiocaps/train/text.json', 'data/audiocaps/val/text.json', 'data/audiocaps/test/text.json']:
    """
    for s in ['data/newdata_audiocaps/train_basesed_fullrules_modified_delconj_text.json',
            'data/newdata_audiocaps/val_basesed_modified_delconj_text.json',
            'data/newdata_audiocaps/test_basesed_modified_delconj_text.json']:
    """
    for s in ['data/clotho_v2/dev/text.json', 'data/clotho_v2/val/text.json', 'data/clotho_v2/eval/text.json']:
    #for s in ['data/audiocaps/train/timelabel_text.json', 'data/audiocaps/val/timelabel_text.json', 'data/audiocaps/test/timelabel_text.json']:
        print(s)
        c = process(s)
        print(c)
        caps += c
    print(caps, (caps[2]+caps[3])/caps.sum()*100, caps[2]+caps[3], caps.sum())

if __name__ == '__main__':
    fire.Fire(process_all)
