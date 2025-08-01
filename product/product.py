import json
import os
from typing import Any, List, Dict


def join_with_and(values: List[str]) -> str:
    """Joins a list of strings into a single string separated by 'و'."""
    return ' و '.join(values)

def parse_data(data: Dict) -> Dict:
    """function to extract a specific key's information."""
    final = {}
    final['url'] = data['seo']['open_graph']['url']
    final["id"] = data["product"]["id"]

    final["title_fa"] = data["product"]["title_fa"]
    final["title_en"] = data["product"]["title_en"]

    final["properties"] = data["product"]["properties"]
    final["attributes"] = {}
    try:
        for attribute in data["product"]["review"]["attributes"]:
            final["attributes"][attribute["title"]] = join_with_and(attribute["values"])

        for specification in data["product"]["specifications"]:
            for attribute in specification["attributes"]:
                if not attribute["title"] in final["attributes"].keys():
                    final["attributes"][attribute["title"]] = join_with_and(attribute["values"])
    except:
        pass

    final["seller_count"] = 0
    final["seller_info"] = {}
    try:
        for variant in data["product"]["variants"]:
            final["seller_info"][variant['seller']["id"]] = {
                "seller_name": variant['seller']['title'],
                "selling_price": variant["price"]["selling_price"],
                "rrp_price": variant["price"]["rrp_price"],
                "discount_percent": variant["price"]["discount_percent"],
            }
        final["seller_count"] = len(final['seller_info'].keys())
    except:
        pass

    final["image_main"] = data["product"]["images"]["main"]["url"][0]
    final["image_list"] = []
    try:
        images = data["product"]["images"]["list"]
        for image in images:
            final["image_list"].append(image["url"][0])
    except:
        pass

    final["brand_id"] = data["product"]["brand"]["id"]
    final["brand_en"] = data["product"]["brand"]["title_en"]
    final["brand_fa"] = data["product"]["brand"]["title_fa"]

    final["description"] = data["product"]["expert_reviews"]["description"]
    final["short_review"] = data["product"]["expert_reviews"]["short_review"]

    final["advantages"] = []
    try:
        for advantage in data["product"]["pros_and_cons"]["advantages"].values():
            final["advantages"].append(advantage)
    except:
        pass

    final["disadvantages"] = []
    try:
        for disadvantage in data["product"]["pros_and_cons"]["disadvantages"].values():
            final["disadvantages"].append(disadvantage)
    except:
        pass

    return final

if __name__ == "__main__":
    for i in range(145):
        print(f"--------------- [FILE] products_{i}.json ---------------")
        file_path = f'../all_data/products_{i}.json'
        with open(file_path, 'r') as f:
            data = json.load(f)
        parsed_list = []
        inactive_counter = 0
        for d in data:
            try:
                if "is_inactive" in d['data']['product'].keys():
                    inactive_counter += 1
                else:
                    parsed_data = parse_data(d['data'])
                    parsed_list.append(parsed_data)
            except Exception as e1:
                print(f'[Error] for parsed_data: {parsed_data['id']}')

        file_path = os.path.join("./parsed", f"products_{i}_parsed.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(parsed_list, f, indent=4, ensure_ascii=False)
        print(f'[{i}] len data: {len(data)}')
        print(f"[{i}] len parsed_list: {len(parsed_list)}")
        print(f"[{i}] len inactive_counter: {inactive_counter}")