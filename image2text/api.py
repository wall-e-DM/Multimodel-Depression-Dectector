from PIL import Image
from typing import List


# mapping
{
  0:"not depression",
  1:"moderate",
  2:"severe"
}


def get_clip_output(img_url,model,processor) -> List[float]:

    # Load image from URL
    # image = Image.open(requests.get(img_url, stream=True).raw)
    image = Image.open(img_url).convert("RGB")

    # Process image and text inputs for CLIP model
    # inputs = processor(text=["not depression","moderate","severe"], images=image, return_tensors="pt", padding=True)
    
    ##### Try to make the labels clear. Pangyu Li - 05.01 11:27
    ##### cute. I guess this is the magic of Prompt engineering :) Zihan Zhou - 05.02 20:41
    inputs = processor(text=["not depression","moderate depression","severe depression"], images=image, return_tensors="pt", padding=True)
    print("------------------init------------------")
    # Run CLIP model on inputs
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    print("------------------end------------------")
    # Convert probabilities to a list and return
    return probs.detach().numpy()

# print(get_clip_output("./images/weirdman.png")) 
# print(get_clip_output("./images/someone.png")) 
# print(get_clip_output("../pipeline/images/not_depression/not-2.jpg"))