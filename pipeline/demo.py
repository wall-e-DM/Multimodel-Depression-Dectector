############################ Resolve import Path ############################
import os
import sys
# from image2text.api import get_clip_output
root_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(root_path)
############################  import modals ############################
from image2text.api import get_clip_output
from utils import get_ensemble_ouput, modality_fusion
############################  get the output ############################
# get the clip output
clip_output = get_clip_output("./images/weirdman.png")

# zip it, it looks like a shit
text_str = "There is no way this shitty condition cant be cured right? : i've been this way for 8 years, yet i still believe theres a cure somehow. in fact, the damage has already been done. am i silly for thinking by brain could go back to normal after experiencing no hapiness for 8 years? theres no way we're gonna live our whole lives like this right? this cant be real. how could this be real? whtas the point of life if we're gonna be like this? i literally only play video games all day. i have no life, no friends, no job, pushed all my family away. there must be a cure right? life is hapiness right? i dont understand. i dont want to accept im going to commit suicide one day. do people realise how fucking hard it must be to commit suicide? i dont want to get run over by a train. i dont want a bullet destroying my brain. but jesus christ, i also dont want this hell of an existance that people dont even believe exists. people doubt me when i say i experience no hapiness, but its true. im a shell of my former self, im not making it up. my old friends told me to my face, ''you used to be so cool, now you're just lame'' how can i go on living? how can i accept being a fucking boring and bored piece of trash doing nothing but playing video games all day? exercise doesnt work, forcing myself to socialise doesnt work. all that i can do is keep playing video games because its better than literally doing nothing. i cant fucking cure myself. i fucking cant. what the fuck do i do if i cant get better? theres no way this is existance right? just having an uncurable condition and suffer until u die? therapists are useless, medication has been useless so far, but i just cant accept my personality and passion is gone forever."
# get the ensemable output
text_classifier_output = get_ensemble_ouput(text_str)

# unzip it to scrutinize it
# print(clip_output)
# print(text_classifier_output)
############################  modality fusion ############################
final_output = modality_fusion(clip_output, text_classifier_output)
print(final_output)