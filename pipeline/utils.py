from typing import List
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from model import Model
from transformers import AutoTokenizer, AutoModelForSequenceClassification,\
                         AdamW, get_scheduler  
import numpy as np
from torch.utils.data import DataLoader
from dataset import DepressDataset
from vad_score import _get_vad_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc

############################  Hyper parameter  ############################
# const area, not a good impl, but i guess it is enough for our data mining cw

DROPOUT = 0.1
HIDDEN = 512
MODEL1 = 'cardiffnlp/twitter-roberta-base-sentiment'
MODEL1_NAME = 'twitter-roberta-base-sentiment_0.5374'
MODEL2 = 'google/electra-base-discriminator'
MODEL2_NAME = 'google-electra-base-discriminator_0.5513'



############################  modality fusion ############################
def modality_fusion(clip_output: np.ndarray, text_output: np.ndarray) -> List[float]:
    # calculate confidence
    confidence_clip = np.max(clip_output)
    confidence_text = np.max(text_output)
    
    # weight the arrays
    weight_clip = confidence_clip / (confidence_clip + confidence_text)
    weight_text = confidence_text / (confidence_clip + confidence_text)
    
    # elementwise ops 
    final_output = list(weight_clip * clip_output + weight_text * text_output)
    print(final_output[0].tolist())
    return final_output[0].tolist()
############################  ensemble eval ############################
# avoid re-import

def get_ensemble_ouput(
    text:str,
    tokenizer1,
    tokenizer2,
    model1,
    model2,
    device,
    config
):
    # basic config
    
    with torch.no_grad():
        input_text1 = tokenizer1(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        input_text2 = tokenizer2(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        # get vad score
        _vad_score = _get_vad_score(text[0])
        vad_score  = torch.unsqueeze(_vad_score, 0)
        predicted_output1, _, _ = model1(vad_score=vad_score.to(device), **input_text1)
        predicted_output2, _, _ = model2(vad_score=vad_score.to(device), **input_text2)
        predicated_output_tmp = torch.add(predicted_output1, predicted_output2) / 2
        sf = F.softmax(predicated_output_tmp, dim=1)

    return sf.cpu().numpy()[0]
############################  groundtruth_generator ############################
def groundtruth_generator(num:int):
    groud_truth = [] # [(label,image_url,text)...]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dataloader = prepare_data(num)
    index_mapper = {
        0 : "not depression",
        1 : "moderate",
        2 : "severe"
    }
    not_depression_cnt,moderate_cnt,severe_cnt = 1,1,1
    for data in dataloader:
        pid, text, vad_score,label= data[0][0],list(data[0]), data[1].to(device),data[2]
        if label.item() == 0 and not_depression_cnt <= num:
            # assign a picture to this text, image url is like /images/not-001.jpg
            groud_truth.append((label.item(),f"./images/not_depression/not-{not_depression_cnt}.jpg",text))
            not_depression_cnt += 1
        elif label.item() == 1 and moderate_cnt <= num:
            groud_truth.append((label.item(),f"./images/moderate/moderate-{moderate_cnt}.jpg",text))
            moderate_cnt += 1
        elif label.item() == 2 and severe_cnt <= num:
            groud_truth.append((label.item(),f"./images/severe/severe-{severe_cnt}.jpg",text))
            severe_cnt += 1
        else:
            continue
    return groud_truth

"""
    从twiter.txt中读取数据
"""
def twiter_loader():
    print("---Using Twiter Set---")
    ground_truth = []
    with open("./twiter.txt", "r") as f:
        f = open("./twiter.txt", "r")
        header = f.readline().strip().split(",\t")
        line = f.readline()
        count = 0
        while line:
            line = f.readline()
            data = line.strip().split(",", 2)
            if len(data) < 3:
                continue
            # 因为twiter数据集的文字部分都是只有一段，这里就直接处理成都是长度为1的list来存
            data[2] = [data[2].strip()[2:-2].strip()]
            ground_truth.append((int(data[0]), data[1], data[2]))
            count += 1
        print(f"---Total {count} Datas---")
    return ground_truth


################################## prepare data ##################################
def prepare_data(mages_per_folder,):
    # this is the bad chocie for coding
    # tmp change
    path = "../data/test_np.tsv"
    mode = "test"
    data = DepressDataset(path, mode)
    dataloader = DataLoader(data, batch_size=1, shuffle=True)
    return dataloader

def test_ground_truth_generator(): # side effects: append the result to the ensemble.txt
    # with open("twiter.txt", "w") as f:
    with open("ensemble.txt", "w") as f:
        f.write("label,\timage_url,\t text\n")
        for label,image_url, text in groundtruth_generator():
            f.write(f"{label},{image_url}, {text}\n")
        
# test_ground_truth_mapper()
# 返回一个 [(label,image_url,text)...]

"""
    将三元label二元化
    根据主元:
        将ground_labels转化成一维数组, 只有1\0二元分类
        将predict_socres转化成一维数组, 只有socres
"""
def binarize(labels:list, pivot:str, ty:str):
    binary_list = []
    if ty=='P':
        if pivot == 'N':
            for score in labels:
                binary_list.append(score[0])
        elif pivot == 'M':
            for score in labels:
                binary_list.append(score[1])
        elif pivot == 'S':
            for score in labels:
                binary_list.append(score[2])
        return binary_list
    elif ty=='G':
        if pivot == 'N':
            for label in labels:
                binary_list.append(1 if label == 0 else 0)
        elif pivot == 'M':
            for label in labels:
                binary_list.append(1 if label == 1 else 0)
        elif pivot == 'S':
            for label in labels:
                binary_list.append(1 if label == 2 else 0)
        return binary_list

"""
    三分类ROC作图
    输入:
        ground_labels 样本真实标签, 是一个n*1的list, 取值是0/1/2
        pred_scores  样本模型输出scores, 是一个n*3的list, 每个item都是3个score表示
"""
def drawROC(ground_labels:list, pred_scores:list, name:str):
    plt.figure(figsize=(10, 10))
    plt.title("ROC-" + name)
    colors = ['palegreen', 'gold', 'coral']
    lagends= ['Not Depressive', 'Moderate Depressive', 'Severe Depressive']
    _fpr, _tpr = [], []
    a = 0
    for i, pivot in zip(range(3), ['N', 'M', 'S']):
        ground_binary = binarize(ground_labels, pivot, 'G')
        text_binary = binarize(pred_scores, pivot, 'P')
        fpr, tpr, thresholds = roc_curve(ground_binary, text_binary)
        plt.plot(fpr, tpr, marker = 'o', label=lagends[i], linestyle='--', color=colors[i])
        _fpr.append(fpr)
        _tpr.append(tpr)
    # 画一下随机线
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', label='Random', alpha=.8)
    # 算一下AUC
    for i in range(3):
        a += auc(_fpr[i], _tpr[i])
    a /= 3
    # 画一下AUC
    plt.text(0.5, 0.3, f"Average AUC={a:.3f}", fontsize=17)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc='lower right')
    plt.savefig(f"./result/ROC-{name}-{timestamp}.png")