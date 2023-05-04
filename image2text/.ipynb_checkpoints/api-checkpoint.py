from PIL import Image
import requests
from typing import List
from transformers import CLIPProcessor, CLIPModel

# mapping
{
  0:"not depression",
  1:"moderate",
  2:"severe"
}

def get_clip_output(img_url) -> List[float]:
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
      # Load image from URL
    # image = Image.open(requests.get(img_url, stream=True).raw)
    image = Image.open(img_url).convert("RGB")

    # Process image and text inputs for CLIP model
    # inputs = processor(text=["not depression","moderate","severe"], images=image, return_tensors="pt", padding=True)
    
    ##### Try to make the labels clear. Pangyu Li - 05.01 11:27
    inputs = processor(text=["not depression","moderate depression","severe depression"], images=image, return_tensors="pt", padding=True)
    
    # Run CLIP model on inputs
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    # Convert probabilities to a list and return
    return probs.detach().numpy()

# print(get_clip_output("./images/weirdman.png")) 
# print(get_clip_output("./images/someone.png")) 
print(get_clip_output("../pipeline/images/not_depression/not-2.jpg"))