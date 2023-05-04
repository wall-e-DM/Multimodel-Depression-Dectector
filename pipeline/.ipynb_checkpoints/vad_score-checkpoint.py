import nltk
import torch
from nltk.sentiment.vader import SentimentIntensityAnalyzer
def __init_nltk():
    nltk.download('vader_lexicon')

# return a vad score  torch.tensor(vad_score = [neg, neu, pos, compound])
def _get_vad_score(text_input)-> torch.tensor:
    sid = SentimentIntensityAnalyzer()
    vad_score = sid.polarity_scores(text_input)
    return torch.tensor([vad_score['neg'], vad_score['neu'], vad_score['pos'], vad_score['compound']])
    

