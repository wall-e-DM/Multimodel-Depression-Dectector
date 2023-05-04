import os
import sys
# from image2text.api import get_clip_output

root_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(root_path)

from image2text.api import get_clip_output
