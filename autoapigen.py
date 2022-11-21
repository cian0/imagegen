import io
import requests
from PIL import Image
import base64
# import IPython.display
import json
import matplotlib
import base64
import sys

encoded = base64.b64encode(open("/home/ian/Pictures/rehtsygg/training_images/20210720_042050.jpg", "rb").read()) #change the directory and
# image name to suit your needs
encodedString=str(encoded, encoding='utf-8')
GoodEncoded='data:image/png;base64,' + encodedString
payload = {
    "init_images": [
        GoodEncoded
    ] ,
    "prompt": "human person rehts",
    "steps": 50,
    "sampler_index": "Euler"
}
payloadJson = json.dumps( payload)
resp = requests.post(url="http://127.0.0.1:7860/sdapi/v1/img2img", data=payloadJson).json()
if resp.get("images") is None:
    print("Error, got this returned:")
    print(resp)
else:
    for i in resp['images']:
        img = Image.open(io.BytesIO(base64.b64decode(i)))
        # display(img)
        im1 = img.save("/home/ian/Pictures/output.png") #change the directory and image name
        # to suit your needs
        ## ignore the stuff below here you won't be interested in this likely
        # images = [Image.open(x) for x in ['D:\PreImageDirectory\geeks1.png',
        # 'D:\PreImageDirectory\geeks1.png']]
        # widths, heights = zip(*(i.size for i in images))
        # total_width = sum(widths)
        # max_height = max(heights)
        # new_im = Image.new('RGB', (total_width, max_height))
        # x_offset = 0
        # for im in images:
        #     new_im.paste(im, (x_offset,0))
        #     x_offset += im.size[0]
        #     new_im.save('D:/ImageDirectory/test2.jpg')
####