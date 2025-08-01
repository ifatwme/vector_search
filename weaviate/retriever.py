import weaviate

import torch
from transformers import CLIPVisionModel, RobertaModel, AutoTokenizer, CLIPFeatureExtractor
from PIL import Image


IMAGE_EMBEDDING_MODEL = "SajjadAyoubi/clip-fa-vision"
TEXT_EMBEDDING_MODEL = "SajjadAyoubi/clip-fa-text"

vision_encoder = CLIPVisionModel.from_pretrained(IMAGE_EMBEDDING_MODEL)
preprocessor = CLIPFeatureExtractor.from_pretrained(IMAGE_EMBEDDING_MODEL)
text_encoder = RobertaModel.from_pretrained(TEXT_EMBEDDING_MODEL)
tokenizer = AutoTokenizer.from_pretrained(TEXT_EMBEDDING_MODEL)


# client = weaviate.Client("http://localhost:8080")
client = weaviate.connect_to_local()

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
    
def search_text_text(text):
    return client.collections.get("Product").query.near_vector(get_text_vector(text), target_vector="text_vector", limit=10)


def search_text_image(text):
    return client.collections.get("Product").query.near_vector(get_text_vector(text), target_vector="image_vector", limit=10)


def search_image_text(image):
    return client.collections.get("Product").query.near_vector(get_image_vector(image), target_vector="text_vector", limit=10)


def search_image_image(image):
    return client.collections.get("Product").query.near_vector(get_image_vector(image), target_vector="text_vector", limit=10)


# Example usage: search_text_against_images
if __name__ == "__main__":
    # query = "modern leather boots for men"
    # matches = search_text_against_images(query)
    # for i, product in enumerate(matches):
    #     print(f"{i+1}. {product['title']}")

# Example usage: search_by_text
# if __name__ == "__main__":
#     query = "Red waterproof hiking backpack with multiple compartments"
#     matches = search_by_text(query)
#     for i, product in enumerate(matches):
#         print(f"{i+1}. {product['title']}")
    
    try:
        query_vector = get_text_vector("لباس مردانه رسمی")
        # query_vector = get_image_vector('2185924_0.jpg')

        result = client.collections.get("Product").query.near_vector(query_vector, target_vector="image_vector", limit=10)
        for obj in result.objects:
            print(f"PID: {obj.properties['pid']}")
            print(f"Title: {obj.properties['title']}")
            print(f"Description: {obj.properties['description']}")
            print("-" * 40)
    finally:
        client.close()
    
    # text="یه چیزی"
    # image = Image.open('8354443_0.jpg')
    # text_embedding = text_encoder.encode(text, convert_to_tensor=True)
    # image_embedding = vision_encoder(**preprocessor(image, 
    #                                     return_tensors='pt')).pooler_output.squeeze(0)
    # print(text_embedding.shape)
    # print(image_embedding.shape)
    # print(image_embedding.shape == text_embedding.shape)