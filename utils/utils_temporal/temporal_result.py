import json
import os
import numpy as np
import pandas as pd
import fire
import tqdm

class result(object):

    def tag_calculate(self, data):
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
        captions = {}
        for i in range(len(data)):
            audio_id = data[i]["filename"]
            captions[audio_id] = [{"audio_id": audio_id,
                "id": 0,
                "caption": data[i]["caption"]}]
        tokenizer = PTBTokenizer()
        captions = tokenizer.tokenize(captions)
        after_span = ["follow", "followed", "then", "after"]
        while_span = ["while", "and", "when", "with", "as"]

        for audio_idx in tqdm.tqdm(range(len(data)), leave=False, ascii=True):
            audio_id = data[audio_idx]["filename"]
            token_list = captions[audio_id][0].split(' ')
            after_flag, while_flag = 0, 0
            for i in range(len(token_list)):
                if token_list[i] in after_span:
                    after_flag = 2
                if token_list[i] in while_span:
                    while_flag = 1
                if after_flag + while_flag == 3:
                    break
            data[audio_idx]["time_label"] = after_flag + while_flag    
        return data
    
    def get_result(self, pre_file: str, 
            gt_file: str ="data/audiocaps/test/timelabel_text.json", score_path:str="last_scores.txt",
            metric = None):
        if metric == None:
            metric = ['Bleu-4', 'Rouge', 'CIDEr', 'METEOR', 'SPICE', "FenseWoPenalty", "acc", "f1"]#, 'Rouge', 'SPIDEr']
        out = json.load(open(pre_file, "r"))["predictions"]
        out = self.tag_calculate(out)
        path = '/'.join(pre_file.split('/')[:-1]) + '/'
        gt_dict = {}
        gt_data = json.load(open(gt_file, "r"))["audios"]
        for i in gt_data:
            gt_dict[i['audio_id']] = i

        confusion = np.zeros((4,4))
        tp, fn, tn, fp = 0, 0, 0, 0

        for i in range(len(out)):
            gt = gt_dict[out[i]['filename']]

            tag_flag, pre_flag = 0, 0
            temporal_tag = 0
            for j in range(len(gt["captions"])):
               if gt["captions"][j]["time_label"] >= 2:
                   tag_flag = 1
               if gt["captions"][j]["time_label"] >= temporal_tag:
                   temporal_tag = gt_data[i]["captions"][j]["time_label"]
            if "human_time_label" in gt.keys():
                temporal_tag = gt["human_time_label"]
                tag_flag = (temporal_tag >= 2) + 0

            if out[i]['time_label'] >= 2 :
                pre_flag = 1

            confusion[temporal_tag][out[i]["time_label"]] += 1
            if tag_flag:
               if pre_flag:
                  tp += 1
               else:
                  fn += 1
            else:
               if pre_flag:
                  fp += 1
               else:
                  tn += 1
            out[i]["gt_vs_pre"] = str(tag_flag) + str(pre_flag)
        f1 = 2 * tp / (2 * tp + fp + fn)
        acc = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp+ fp)
        recall = tp / (tp + fn)
        print("recall:", recall*100, "precision:", precision*100)
        #print(f1, 2*precision*recall/(precision+recall))
        print(tp, tn, fn, fp)
        #print("confusion:", confusion)
        #json.dump({"predictions": out}, open(path + "last_output_timelabel_result.json", "w"), indent=4)  
        scores_dict = {"acc":acc, "f1":f1}
        with open(pre_file[:-16] + score_path, "r") as f:
            for line in  f.readlines():
                line = line.strip("\n").split(" ")
                if len(line)>1:
                    scores_dict[line[0][:-1]] = line[-1]
        if 'FenseWoPenalty' not in scores_dict.keys() and 'Fense' in scores_dict.keys():
            scores_dict['FenseWoPenalty'] = scores_dict['Fense']
        out_score = [str(round(float(scores_dict[k])*100, 1)) for k in metric]
        print('&',' &'.join(out_score))
        #print(out)
        return out_score

    def get_result_all(self, path:str, gt_file:str="data/audiocaps/test/timelabel_text.json", 
            out:str='average_score', type:int = 1): #seeds:list =['1000']):#['1','10','100','1000']):
        if type == 1:
            seeds = ['1000']
        else:
            seeds = ['1','10','100','1000']

        #metric = ['Bleu-4', 'Rouge', 'CIDEr', 'METEOR', 'SPICE', "FenseWoPenalty", "Fense", "acc", "f1"]#, 'Rouge', 'SPIDEr']
        metric = ['Bleu-4', 'Rouge', 'CIDEr', 'METEOR', 'SPICE', "FenseWoPenalty", "acc", "f1"]#, 'Rouge', 'SPIDEr']
        metrics_num = len(metric)
        scores = np.zeros((metrics_num, len(seeds)))
        for idx, seed in enumerate(seeds):
            return_list = self.get_result( path + "seed_" + seed + "/", gt_file, metric)
            print(return_list)
            for j in range(metrics_num):
                scores[j][idx] = return_list[j]

        scores *= 100
        output = []
        if type == 1:
            for j in range(metrics_num):
                #output.append(round(scores[j].mean(),1))
                #output.append("{}($\pm${})".format(round(scores[j].mean(),1), round(scores[j].var(), 1)))
                output.append(f"{round(scores[j].mean(),1)}")
        else:
            for j in range(metrics_num):
                #output.append(round(scores[j].mean(),1))
                #output.append("{}($\pm${})".format(round(scores[j].mean(),1), round(scores[j].var(), 1)))
                output.append("{}({})".format(round(scores[j].mean(),1), round(scores[j].var(), 1)))
                #output.append(" {} ".format(round(scores[j].mean(),1)))
        print('&',' &'.join(output))
        print(' '.join(output))
        
        with open(path + out, "w") as f:
            for i in seeds:
                f.write("seeds: {}\n".format(i))
            for j in range(metrics_num): 
                f.write("{}: {}\n".format(metric[j], output[j]))

if __name__ == '__main__':
    fire.Fire(result)
