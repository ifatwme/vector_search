import json
import os
import requests
from PIL import Image

def download_image(url: str, save_dir: str, image_id: str) -> str:
    """
    Downloads an image from the given URL and saves it in the specified
    directory with the given ID as filename.
    """
    ext = os.path.splitext(url.split("?")[0])[1].lower()
    temp_path = os.path.join(save_dir, f"{image_id}{ext}")
    final_path = os.path.join(save_dir, f"{image_id}.jpg")

    if not os.path.isfile(temp_path):
        os.makedirs(save_dir, exist_ok=True)

        r = requests.get(url, stream=True)
        if r.status_code != 200:
            return None

        with open(temp_path, 'wb') as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)
    if os.path.isfile(temp_path):
        if ext == ".jpg":
            return temp_path

        with Image.open(temp_path) as img:
            img.convert("RGB").save(final_path, "JPEG")
        os.remove(temp_path)
        return final_path
    else:
        return None

with open('../parsed/products_143_parsed.json', 'r') as json_file:
    data = json.load(json_file)
    json_file.close()

titles = []
for i, product in enumerate(data):
    i = i + 1
    print(product['title_fa'])
    titles.append(product['title_fa'])

    with open(f'./text_test/{i}_{product['id']}.txt', 'w') as f:
        f.write(product['title_fa'])
        f.close()

    download_image(product['image_main'], f'./image_test/', f"{i}_{product['id']}")
    if i == 100:
        break
    

# with open('./titles.txt', 'w') as f:
#     f.write(str(titles))
#     f.close()