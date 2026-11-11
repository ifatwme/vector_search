# python documentation: https://weaviate.io/developers/weaviate
# frequent examples: https://weaviate.io/developers/weaviate/client-libraries/python#code-examples--resources
# https://weaviate.io/developers/weaviate/api/rest
import weaviate
from weaviate.classes.config import Configure, DataType, Property

import numpy as np
import torch
from transformers import CLIPVisionModel, RobertaModel, AutoTokenizer, CLIPFeatureExtractor

from PIL import Image
import base64
import json
import os
import requests

# client = weaviate.Client("http://localhost:8080")
client = weaviate.connect_to_local()

MODEL_NAME = "openai/clip-vit-base-patch32"
IMAGE_EMBEDDING_MODEL = "SajjadAyoubi/clip-fa-vision"
TEXT_EMBEDDING_MODEL = "SajjadAyoubi/clip-fa-text"

vision_encoder = CLIPVisionModel.from_pretrained(IMAGE_EMBEDDING_MODEL)
preprocessor = CLIPFeatureExtractor.from_pretrained(IMAGE_EMBEDDING_MODEL)
text_encoder = RobertaModel.from_pretrained(TEXT_EMBEDDING_MODEL)
tokenizer = AutoTokenizer.from_pretrained(TEXT_EMBEDDING_MODEL)

def get_text_vector(text):
    with torch.no_grad():
        output = text_encoder(**tokenizer(text, return_tensors='pt', truncation=True, padding=True))
    return output.pooler_output.squeeze(0).tolist()

def get_image_vector(image_path):
    image = Image.open(image_path).convert("RGB")
    with torch.no_grad():
        inputs = preprocessor(image, return_tensors='pt')
        output = vision_encoder(**inputs)
        out = output.pooler_output.squeeze(0)
        return out.tolist() 

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


def image_to_base64(image_path):
    """
    Reads an image from the given file path and
    encodes it to a Base64 string.
    """
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
        encoded_string = base64.b64encode(image_data).decode('utf-8')
    return encoded_string

def import_data(json_path):
    with open(json_path, 'r') as json_file:
        data_object = json.load(json_file)
        json_file.close()

    print(client.get_meta())
    print(client.collections.list_all(simple=True))
    if "Product" not in client.collections.list_all().keys():
        collection = client.collections.create(
            "Product",
            description='A record that stores only the embedding and a reference to external data. Images are referenced via MinIO paths.',
            properties=[
                Property(name="pid",data_type=DataType.INT),
                Property(name="title",data_type=DataType.TEXT),
                Property(name="description",data_type=DataType.TEXT),
                Property(name="attributes",data_type=DataType.TEXT),
                Property(name="image",data_type=DataType.TEXT), # image_main
                Property(name="price",data_type=DataType.TEXT),
                Property(name="text",data_type=DataType.TEXT),
                Property(name="image_vector", data_type=DataType.NUMBER_ARRAY),
                Property(name="text_vector", data_type=DataType.NUMBER_ARRAY),

            ],
            vectorizer_config=[
                Configure.NamedVectors.none(name="text_vector"),
                Configure.NamedVectors.none(name="image_vector"),
            ]
        )
        print(f"collection created: \n {collection}")
    else:
        collection = client.collections.get('Product')

    with collection.batch.dynamic() as batch:
        for src_obj in data_object:
            print(f"---{src_obj['id']}---")
            image_path = f'../product/images/{src_obj["id"]}/{src_obj["id"]}_0.jpg'
            title = f"{src_obj["title_fa"]} {src_obj["title_en"]}"
            download_flag = download_image(src_obj['image_main'], f'../product/images/{src_obj['id']}', f"{src_obj['id']}_0")
            if download_flag:
                text_vector = get_text_vector(title)
                image_vector = get_image_vector(image_path)
                
                weaviate_obj = {
                    "pid": src_obj['id'],
                    "title": title,
                    "description": f"{src_obj['description']} {src_obj['short_review']}",
                    "attributes": f"{src_obj["attributes"]} {src_obj['properties']}",
                    "price": f"{src_obj['seller_info']}",
                    "image": f'../product/images/{src_obj["id"]}/{src_obj["id"]}_0.jpg',
                    "text_vector": text_vector.tolist() if isinstance(text_vector, np.ndarray) else text_vector,
                    "image_vector": image_vector.tolist() if isinstance(image_vector, np.ndarray) else image_vector,
                    "url": src_obj['url'],
                }

                uuid = batch.add_object(weaviate_obj, vector={"text_vector": weaviate_obj["text_vector"], "image_vector": weaviate_obj["image_vector"]})
                print(f"[DONE] data inserted")
            else:
                print("[ERROR] failed to download")

if __name__ == "__main__":
    # single example, all 0 to 144
    # json_path = "../product/parsed/products_0_parsed.json"
    # import_data(json_path)
    try:
        for i in range(1,10):
            print(f"------products_{i}_parsed.json-------")
            import_data(f"../product/parsed/products_{i}_parsed.json")
    finally:
        client.close()
