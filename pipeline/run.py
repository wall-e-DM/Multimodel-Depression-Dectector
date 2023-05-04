from flask import Flask, jsonify, request
from flask_restful import Resource, Api
from flask_cors import CORS, cross_origin
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
from transformers import CLIPProcessor, CLIPModel
import seaborn as sns
# from image2text.api import get_clip_output
root_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(root_path)
############################  import necessities ###############################
from utils import get_ensemble_ouput, groundtruth_generator, modality_fusion
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
groud_list = groundtruth_generator(image_per_foler) #这个最重要
############################### multi-modal exp ###############################
def clip_text_multi_modal(image_url: str, text: str):
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
    print(f"pred_label: {str(final_output)}")
    return final_output

############################### flask #################################
app = Flask(__name__)
api = Api(app)
cors = CORS(app,supports_credentials=True)

class Test(Resource):
    def get(self):
        return jsonify({'message': 'Hello, World!'})

class DepressionApi(Resource):
    def get(self):
        #这里从url里拿到text和image的url
        image_url = request.args.get('image_url')
        text = request.args.get('text')
        #调用clip_text_multi_modal
        pred_label = clip_text_multi_modal(image_url, text)
        response = jsonify({'message': str(pred_label)})
        return response

class ImageApi(Resource):
    def post(self):
        #这里处理传文件的操作
        #flask处理文件的方法
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            url = f"./flask_img/{timestamp}.jpg"
            uploaded_file.save(url)
            response = jsonify({'status': 'done', 'message': url})
            return response
        else:
            return jsonify({'message': 'No file is uploaded'})

api.add_resource(DepressionApi, '/depression')
api.add_resource(ImageApi, '/image')
api.add_resource(Test, '/test')

if __name__ == '__main__':
    app.run(debug=True)