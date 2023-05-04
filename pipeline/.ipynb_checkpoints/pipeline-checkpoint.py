################### Define the pipline and clarification ###################


# 我们定义一个社交场景为 
# {
#    (image,post_text),
#    (image,post_text),
#    .......
#  }
# 1. 100个图片+text 得到指标 
# 2. 100个text-only (roberta+electra ensenmble) 得到指标 and clip only 得到指标
# 3. 100个text-only + fine-tuning chatGPT 得到指标
 

############################ Resolve import Path ###############################
import os
import sys
import time
from typing import List
import numpy as np
import torch
from model import Model
from transformers import AutoTokenizer, AutoModelForSequenceClassification,\
                         AdamW, get_scheduler  
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import CLIPProcessor, CLIPModel
# from image2text.api import get_clip_output
root_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(root_path)
############################  import necessities ###############################
from utils import get_ensemble_ouput, groundtruth_generator, modality_fusion, drawROC
from image2text.api import get_clip_output
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
######################### config and hyper-parameters ###########################
DROPOUT = 0.1
HIDDEN = 512
MODEL1 = 'cardiffnlp/twitter-roberta-base-sentiment'
MODEL1_NAME = 'twitter-roberta-base-sentiment_0.5374'
MODEL2 = 'google/electra-base-discriminator'
MODEL2_NAME = 'google-electra-base-discriminator_0.5513'

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
config = {
    'dropout': DROPOUT,
    'hidden': HIDDEN
}
# model init
model1 = Model(MODEL1, config).to(device)
model2 = Model(MODEL2, config).to(device)
tokenizer1 = AutoTokenizer.from_pretrained(MODEL1)
tokenizer2 = AutoTokenizer.from_pretrained(MODEL2)
model1.load_state_dict(torch.load(f"../model/{MODEL1_NAME}.pt"))
model2.load_state_dict(torch.load(f"../model/{MODEL2_NAME}.pt"))

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
################################# Exp functions #################################
image_per_foler = 25
groud_list = groundtruth_generator(image_per_foler)
############################### multi-modal exp ###############################
def exp_clip_text_multi_modal(groud: list):
    ground_labels = []
    pred_labels = []
    pred_scores = []
    # 1. 100 images + text with metrics
    # clip_output = get_clip_output("./images/weirdman.png")
     # [(label,image_url,text)...]
    for item in groud_list:
        ground_label, image_url, text = item[0], item[1], item[2]
        clip_output = get_clip_output(image_url,model,processor)
        text_output = get_ensemble_ouput(
            text,
            tokenizer1,
            tokenizer2,
            model1,
            model2,
            device,
            config
        )
        final_output = modality_fusion(clip_output, text_output)
        # use argmax to get the pred_label
        pred_label = np.argmax(final_output)
        ground_labels.append(ground_label)
        pred_labels.append(pred_label)
        pred_scores.append(final_output)
    # calculate the precision, recall, f1
    precision = precision_score(ground_labels, pred_labels, average='macro')
    recall = recall_score(ground_labels, pred_labels, average='macro')
    f1 = f1_score(ground_labels, pred_labels, average='macro')
    accuracy = accuracy_score(ground_labels, pred_labels)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    # draw the plot
    cm = confusion_matrix(ground_labels, pred_labels)
    
    drawROC(ground_labels, pred_scores, 'multimodal')

    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, cmap="Blues")
    plt.title("Confusion matrix")
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.savefig(f"./result/confusion_matrix-mutlti-{timestamp}.png")

    with open("./log_result_truth.txt", "a") as f:
        f.write("-----------------------------\n")
        f.write("exp_clip_text_mutil_modal\n")
        f.write("timestamp: {}\n".format(timestamp))
        f.write(f"precision: {precision} recall: {recall} f1: {f1} accuracy: {accuracy}\n")
        f.write("-----------------------------\n")
    
def exp_ensemble_model(groud:List):
    ground_labels = []
    pred_labels = []
    pred_scores = []
    for item in groud_list:
        ground_label,text = item[0], item[2]
        text_output = get_ensemble_ouput(
            text,
            tokenizer1,
            tokenizer2,
            model1,
            model2,
            device,
            config
        )
        # use argmax to get the pred_label
        pred_label = np.argmax(text_output)
        ground_labels.append(ground_label)
        pred_labels.append(pred_label)
        pred_scores.append(text_output)
    # calculate the precision, recall, f1
    precision = precision_score(ground_labels, pred_labels, average='macro')
    recall = recall_score(ground_labels, pred_labels, average='macro')
    f1 = f1_score(ground_labels, pred_labels, average='macro')
    accuracy = accuracy_score(ground_labels, pred_labels)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    drawROC(ground_labels, pred_scores, 'text')

    # draw the plot
    cm = confusion_matrix(ground_labels, pred_labels)
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, cmap="Blues")
    plt.title("Confusion matrix")
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.savefig(f"./result/confusion_matrix-ensemble-{timestamp}.png")
    # log the result with timestamp
    with open("./log_result_truth.txt", "a") as f:
        f.write("-----------------------------\n")
        f.write("exp_ensemble_model\n")
        f.write("timestamp: {}\n".format(timestamp))
        f.write(f"precision: {precision} recall: {recall} f1: {f1} accuracy: {accuracy}\n")
        f.write("-----------------------------\n")
    
if __name__ == "__main__":
    exp_clip_text_multi_modal(groud_list)
    exp_ensemble_model(groud_list)

