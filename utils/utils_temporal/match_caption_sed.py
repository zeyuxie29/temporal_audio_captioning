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


def process(suffix:str, wo_s: int=0, wo_a: int=0, wo_d: int=0, wo_audioset: int=0, only_test: int=0, all_tag:int=-1,

        after_thres=0.1, while_thres=1):
    total = {"add":0, "del":0, "switch":0, "all":0, "add_sed":0, "del_sed":0}
    vocab = pickle.load(open("data/audiocaps/train/vocab.pkl", "rb"))
    audiosetlabel_to_vocab  = pickle.load(open("data/audiocaps/all/audiosetlabel_to_vocab.pkl", "rb"))

    while_vocab = ['while ', 'and ', 'when ', 'with ', 'as ']
    after_vocab = ['follow ', 'followed by ', 'then ', 'after '] 
    splits = ['train', 'val', 'test']
    if only_test:
        splits = ["test"]
    for split in splits:
        data = json.load(open("data/audiocaps/" + \
            "{}/timelabel_text.json".format(split), "r"))['audios']
        audioset_label = pd.read_json("data/newdata_audiocaps/{}_addinfo.json".format(split))['audios']

        ## SED result
        print('Get SED result')
        edecay = pd.read_csv("sed/output/{}_output.csv".format(split), sep="\t")
        edecay_dict = {}
        for i in range(len(edecay)):
            tmp = edecay.iloc[i]
            if tmp['audio_id'] not in edecay_dict.keys():
                tmp_add = []
                edecay_dict[tmp['audio_id']] = [{'s':tmp['onset'], 'e':tmp['offset'], 'label':tmp['event_label']}]
            else:
                edecay_dict[tmp['audio_id']].append({'s':tmp['onset'], 'e':tmp['offset'], 'label':tmp['event_label']})
        print('Concatenate SED result')
        c = np.zeros((4,4))
        for i in range(len(data)):
            total['all'] += 1
            tag = 'sed_label'
            if data[i]['audio_id'] not in edecay_dict.keys() or len(edecay_dict[data[i]['audio_id']])<=1:
                data[i][tag] = 0
            else:
                tmp = edecay_dict[data[i]['audio_id']]
                after_flag, while_flag = 0, 0
                for j in range(len(tmp)):
                    for k in range(j+1, len(tmp)):
                        if tmp[j]['label'] != tmp[k]['label'] and ((tmp[k]['s'] > tmp[j]['e']) or ((tmp[j]['e']-tmp[k]['s']) < after_thres * min(tmp[j]['e']-tmp[j]['s'], tmp[k]['e']-tmp[k]['s']))):
                            after_flag = 2
                        if tmp[j]['label'] != tmp[k]['label'] and (tmp[j]['e']-tmp[k]['s']) > while_thres * min(tmp[j]['e']-tmp[j]['s'], tmp[k]['e']-tmp[k]['s']):
                            while_flag = 1
                data[i][tag] = after_flag + while_flag
            c[data[i]['captions'][0]['time_label']][data[i]['sed_label']] += 1
        print(c)

        # modify annotation
        c_add, c_del, c_switch, c_add_sed, c_del_sed = 0, 0, 0, 0, 0
        for i in tqdm.tqdm(range(len(data))):
            tem_sed_label = -1
            for j in range(len(data[i]['captions'])):
                # caption 1, sed 2 / caption 2, sed 1
                audiosetlabels = audioset_label[i]['gt_label'].split(',')
                if data[i]['captions'][j]['time_label'] == 1 and data[i]['sed_label'] == 2 and not wo_s:
                    c_switch += 1
                    for idx in range(len(while_vocab)):
                        data[i]['captions'][j]['tokens'] = re.sub(while_vocab[idx], 'then ', data[i]['captions'][j]['tokens'])
                    data[i]['captions'][j]['time_label'] = data[i]['sed_label']
                elif data[i]['captions'][j]['time_label'] == 2 and data[i]['sed_label'] == 1 and not wo_s:
                    c_switch += 1
                    for idx in range(len(after_vocab)):
                        data[i]['captions'][j]['tokens'] = re.sub(after_vocab[idx], 'while ', data[i]['captions'][j]['tokens'])
                    data[i]['captions'][j]['time_label'] = data[i]['sed_label']
                # caption 0, sed 1/2
                elif data[i]['captions'][j]['time_label'] == 0 and data[i]['sed_label'] > 0 and data[i]['sed_label'] <= 2:
                    if len(audiosetlabels) == 1:
                      if split == 'train':
                        if not wo_audioset:
                            tem_sed_label = max(0, tem_sed_label)
                            c_add_sed += 1
                    elif not wo_a:
                        c_add += 1
                        data[i]['captions'][j]['time_label'] = data[i]['sed_label']
                        if data[i]['sed_label'] == 1:
                            data[i]['captions'][j]['tokens'] += ' while ' + audiosetlabel_to_vocab[audiosetlabels[0]]
                        elif data[i]['sed_label'] == 2:
                            data[i]['captions'][j]['tokens'] += ' then ' + audiosetlabel_to_vocab[audiosetlabels[0]]
                
                # caption 1/2, sed 0
                elif data[i]['captions'][j]['time_label'] > 0 and data[i]['captions'][j]['time_label'] <= 2 and data[i]['sed_label'] == 0:
                    if len(audiosetlabels) > 1:
                      if split == 'train':
                        if not wo_audioset:
                            c_del_sed += 1
                            tem_sed_label = max(tem_sed_label, data[i]['captions'][j]['time_label'])
                            #tem_sed_label = max(tem_sed_label, min(3, len(audiosetlabels))) #data[i]['captions'][j]['time_label']
                    elif not wo_d:
                        c_del += 1
                        if audiosetlabel_to_vocab[audiosetlabels[0]] != '':
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
                if split != 'train' and not wo_audioset:
                    if data[i]['sed_label'] > 0 and len(audiosetlabels) == 1:
                        tem_sed_label = 0
                    elif data[i]['sed_label'] < len(audiosetlabels): 
                        tem_sed_label = min(3,len(audiosetlabels))  
            if tem_sed_label > -1:
                data[i]['sed_label'] = tem_sed_label
            if all_tag >=0 :
                data[i]['sed_label'] = all_tag
        print(c_switch, c_add, c_del, c_add_sed, c_del_sed)
        total["switch"] += c_switch
        total["add"] += (c_add + c_add_sed)
        total["add_sed"] += c_add_sed
        total["del"] += (c_del + c_del_sed)
        total["del_sed"] += c_del_sed
        #data = pd.DataFrame(data)
        #json.dump({ "audios": data }, open("data/newdata_audiocaps/{}_{}_text.json".format(split, suffix), "w"), indent=4) 
        #if split == 'test':
        #    for i in range(len(data)):
        #        audiosetlabels = audioset_label[i]['gt_label'].split(',')
        #        if data[i]['sed_label'] > 0 and data[i]['sed_label'] <= 2 and len(audiosetlabels) == 1:
        #            data[i]['sed_label'] = 0
        #    json.dump({ "audios": data }, open("data/newdata_audiocaps/{}_audiosetTag_{}.json".format(split, outfile), "w"), indent=4)
    print(total)
if __name__ == "__main__":
    fire.Fire(process)
