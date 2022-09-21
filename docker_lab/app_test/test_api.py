import requests
from PIL import Image
import io

resp = requests.get('http://127.0.0.1:8000/plot-iris')
file = io.BytesIO(resp.content)
im = Image.open(file)
im.show()