import os
import sys
sys.path.append(os.getcwd())
import json
import pickle
import pandas as pd
import numpy as np
import tqdm
import re
from utils.kw_build_vocab import Vocabulary
import fire
import random

def process(suffix:str="", sed_model:str="cnn8", dataset:str="audiocaps", modify_annotation: int=1, 
        wo_s: int=0, wo_a: int=0, wo_d: int=0, wo_weaklabel: int=0, 
        only_test: int=0, all_tag:int=-1,
        after_thres=0.1, while_thres=1):
    print("sed_model:",sed_model, "dataset:", dataset, "after_thres:", after_thres, "while_thres:", while_thres)
    random.seed(0)
    total = {"add":0, "del":0, "switch":0, "all":0, "add_sed":0, "del_sed":0}
    #vocab = pickle.load(open("data/audiocaps/train/vocab.pkl", "rb"))
    audiosetlabel_to_vocab  = pickle.load(open("data/audiocaps/all/audiosetlabel_to_vocab.pkl", "rb"))
    while_vocab = ['while ', 'and ', 'when ', 'with ', 'as ']
    after_vocab = ['follow ', 'followed by ', 'then ', 'after '] 

    ## SED result
    print('Get SED result')
    save_path = "cnn14_att_finetune_from_weak/" if sed_model == "cnn14" else "cnn8_rnn_finetune_from_weak/"
    filename = "audiocaps_result.csv" if dataset == "audiocaps" else "clotho_result.csv"
    sed_output = "/mnt/lustre/sjtu/home/xnx98/work/SoundEventDetection/audioset_event_detection/experiments/clip_frame_loss/" + save_path + filename
    
    print("sed_output_file:", sed_output)
    descendants = pickle.load(open("data/audioset/descendants.pkl", "rb"))
    strong_to_weak = pickle.load(open("data/audioset/strong_to_weak_with_sbert.pkl", "rb"))
    edecay = pd.read_csv(sed_output, sep="\t")
    edecay_dict = {}
    for i in tqdm.tqdm(range(len(edecay))):
        tmp = edecay.iloc[i]
        label = re.sub(r'[,]', '', tmp['event_label'])
        label = re.sub(r'[ ]', '_', label)
        tmp_add = {'onset':tmp['onset'], 'offset':tmp['offset'], 'label':label}
        if tmp['audio_id'] not in edecay_dict.keys():
            edecay_dict[tmp['audio_id']] = [tmp_add]
        else:
            edecay_dict[tmp['audio_id']].append(tmp_add)
        
        """
        label = strong_to_weak[label]
        tmp_add = {'onset':tmp['onset'], 'offset':tmp['offset'], 'label':label}
        if tmp['audio_id'] not in edecay_dict.keys():
            edecay_dict[tmp['audio_id']] = [tmp_add]
        else:
            tmp_add_flag = 1
            for j in edecay_dict[tmp['audio_id']]:
                if j['label'] == 'ancestor_to_delete':
                    continue
                if tmp_add['label'] in descendants[j['label']]:
                    j['label'] = 'ancestor_to_delete'
                if j['label'] in descendants[tmp_add['label']]:
                    tmp_add_flag = 0
            if tmp_add_flag:
                edecay_dict[tmp['audio_id']].append(tmp_add)
    for k, v in edecay_dict.items():
        edecay_dict[k] = [j for j in v if j["label"] != "ancestor_to_delete"]
    """
    if dataset == "audiocaps": 
        splits = ['train', 'val', 'test']
        if only_test:
            splits = ["test"]
    else:
        splits = ['dev', 'val', 'eval']
        if only_test:
            splits = ["eval"]
    for split in splits:
        text = "timelabel_" if (split == "train" or split == "dev") else ""
        data = json.load(open("data/" + dataset + \
            "/{}/{}text.json".format(split, text), "r"))['audios']

        print('Merge SED result')
        confusion = np.zeros((4,4))
        for i in range(len(data)):
            total['all'] += 1
            if data[i]['audio_id'] not in edecay_dict.keys() or len(edecay_dict[data[i]['audio_id']]) <= 1:
                data[i]['sed_label'] = 0
            else:
                tmp = edecay_dict[data[i]['audio_id']]
                after_flag, while_flag = 0, 0
                for j in range(len(tmp)):
                    for k in range(len(tmp)):
                        if tmp[j]['label'] == tmp[k]['label']:
                            continue
                        min_duration = min(tmp[j]['offset'] - tmp[j]['onset'], tmp[k]['offset'] - tmp[k]['onset'])
                        overlap = tmp[j]['offset'] - tmp[k]['onset']
                        if overlap < after_thres * min_duration:
                            after_flag = 2
                        if tmp[j]['onset'] < tmp[k]['onset'] and overlap > while_thres * min_duration:
                            while_flag = 1
                data[i]['sed_label'] = after_flag + while_flag
            #confusion[data[i]['captions'][0]['time_label']][data[i]['sed_label']] += 1
        print(confusion)
        """
        # modify annotation
        if modify_annotation:
            print("modify_annotation!")
            weak_label_list = pd.read_json("data/newdata_audiocaps/{}_addinfo.json".format(split))['audios']
            c_add, c_del, c_switch, c_add_sed, c_del_sed = 0, 0, 0, 0, 0
            for i in tqdm.tqdm(range(len(data))):
                tem_sed_label = -1
                for j in range(len(data[i]['captions'])):
                    # caption 1, sed 2 / caption 2, sed 1
                    weak_labels = weak_label_list[i]['gt_label'].split(',')
                    if data[i]['captions'][j]['time_label'] == 1 and data[i]['sed_label'] == 2 and not wo_s:
                        c_switch += 1
                        for idx in range(len(while_vocab)):
                            data[i]['captions'][j]['tokens'] = re.sub(while_vocab[idx], after_vocab[random.randint(0, len(after_vocab)-1)], data[i]['captions'][j]['tokens'])
                        data[i]['captions'][j]['time_label'] = data[i]['sed_label']
                    elif data[i]['captions'][j]['time_label'] == 2 and data[i]['sed_label'] == 1 and not wo_s:
                        c_switch += 1
                        for idx in range(len(after_vocab)):
                            data[i]['captions'][j]['tokens'] = re.sub(after_vocab[idx], while_vocab[random.randint(0, len(while_vocab)-1)], data[i]['captions'][j]['tokens'])
                        data[i]['captions'][j]['time_label'] = data[i]['sed_label']
                    # caption 0, sed 1/2
                    elif data[i]['captions'][j]['time_label'] == 0 and data[i]['sed_label'] > 0 and data[i]['sed_label'] <= 2:
                        if len(weak_labels) == 1:
                          if split == 'train':
                            if not wo_weaklabel:
                                tem_sed_label = max(0, tem_sed_label)
                                c_add_sed += 1
                        elif not wo_a:
                            c_add += 1
                            data[i]['captions'][j]['time_label'] = data[i]['sed_label']
                            if data[i]['sed_label'] == 1:
                                data[i]['captions'][j]['tokens'] += while_vocab[random.randint(0, len(while_vocab)-1)] + audiosetlabel_to_vocab[weak_labels[0]]
                            elif data[i]['sed_label'] == 2:
                                data[i]['captions'][j]['tokens'] += after_vocab[random.randint(0, len(after_vocab)-1)] + audiosetlabel_to_vocab[weak_labels[0]]
                    
                    # caption 1/2, sed 0
                    elif data[i]['captions'][j]['time_label'] > 0 and data[i]['captions'][j]['time_label'] <= 2 and data[i]['sed_label'] == 0:
                        if len(weak_labels) > 1:
                          if split == 'train':
                            if not wo_weaklabel:
                                c_del_sed += 1
                                tem_sed_label = max(tem_sed_label, data[i]['captions'][j]['time_label'])
                                #tem_sed_label = max(tem_sed_label, min(3, len(audiosetlabels))) #data[i]['captions'][j]['time_label']
                        elif not wo_d:
                            c_del += 1
                            if audiosetlabel_to_vocab[weak_labels[0]] != '':
                                tmp = data[i]['captions'][j]['tokens'].split(' ')
                                if data[i]['captions'][j]['time_label'] == 1:
                                    for tokens_idx in range(len(tmp)):
                                        if (tmp[tokens_idx] + ' ') in while_vocab:
                                            data[i]['captions'][j]['tokens'] = ' '.join(tmp[:tokens_idx])
                                            break
                                else:
                                    for tokens_idx in range(len(tmp)):
                                        if (tmp[tokens_idx] + ' ') in after_vocab or tmp[tokens_idx] == 'followed':
                                            data[i]['captions'][j]['tokens'] = ' '.join(tmp[:tokens_idx])
                                            break
                                data[i]['captions'][j]['time_label'] = 0
                                #data[i]['captions'][j]['tokens'] = audiosetlabel_to_vocab[audiosetlabels[0]] 
                                #print( data[i]['captions'][j]['tokens'])
                    #if data[i]['captions'][j]['time_label'] == 3 and data[i]['sed_label'] < 3:
                    if split != 'train' and not wo_weaklabel:
                        if data[i]['sed_label'] > 0 and len(weak_labels) == 1:
                            tem_sed_label = 0
                        elif data[i]['sed_label'] < len(weak_labels): 
                            tem_sed_label = min(3,len(weak_labels))  
                if tem_sed_label > -1:
                    data[i]['sed_label'] = tem_sed_label
                if all_tag >= 0:
                    data[i]['sed_label'] = all_tag
            print(c_switch, c_add, c_del, c_add_sed, c_del_sed)
            total["switch"] += c_switch
            total["add"] += (c_add + c_add_sed)
            total["add_sed"] += c_add_sed
            total["del"] += (c_del + c_del_sed)
            total["del_sed"] += c_del_sed
            #data = pd.DataFrame(data)
        """
        json.dump({ "audios": data }, open("data/sedV2/{}{}_{}_{}_text.json".format(save_path, dataset, split, suffix), "w"), indent=4) 
        #if split == 'test':
        #    for i in range(len(data)):
        #        audiosetlabels = audioset_label[i]['gt_label'].split(',')
        #        if data[i]['sed_label'] > 0 and data[i]['sed_label'] <= 2 and len(audiosetlabels) == 1:
        #            data[i]['sed_label'] = 0
        #    json.dump({ "audios": data }, open("data/newdata_audiocaps/{}_audiosetTag_{}.json".format(split, outfile), "w"), indent=4)
    print(total)
if __name__ == "__main__":
    fire.Fire(process)
