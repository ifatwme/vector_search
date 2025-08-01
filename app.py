import streamlit as st

import google.generativeai as genai

import torch
from transformers import CLIPVisionModel, RobertaModel, AutoTokenizer, CLIPFeatureExtractor
from PIL import Image

import weaviate

# Embedding models Configuarions
IMAGE_EMBEDDING_MODEL = "SajjadAyoubi/clip-fa-vision"
TEXT_EMBEDDING_MODEL = "SajjadAyoubi/clip-fa-text"

# Language model Configurations
genai.configure(api_key="AIzaSyBdWBL8dudUTuahhOxRCM107m474i-bjRg")
generation_config = {
    "temperature": 1.5,  # min_value=0.0, max_value=2.0
    "top_p": 1,          # min_value=0.01, max_value=1.0
    "max_output_tokens": 512    # min_value=32, max_value=128
}
system_instruction='''
تو یک ربات فروشنده محصولات دیجیکالا هستی و تو وظیقه داری که با اطلاعاتی درباره محصولات جستجو شده در اختیارت قرار میدهم افراد رو راهنمایی کنی. به سوالات افراد درباره کالا پاسخ بدهی
همیشه سعی کن پاسخ مفید بدهی و درباره هر آنچه که از تو سوال شده نظر بدهی. 
ابتدا رسمی و ربات گونه و کوتاه خودت رو معرفی کن
'''

# Streamlit Configurations
st.set_page_config(
    page_title="Vector Search",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
    <style>
        /* Reduce spacing between checkboxes and sliders */
        div[data-testid="stHorizontalBlock"] {
            gap: 5px !important;
            margin-bottom: 0px !important;
        }
        /* Reduce space below sliders */
        .stSlider {
            padding: 0px !important;
            margin-bottom: 0px !important;
        }
        /* Reduce space around checkboxes */
        .stCheckbox {
            padding: 0px !important;
            margin-bottom: 0px !important;
        }
        /* Reduce space under each row */
        hr {
            margin-top: 0px !important;
            margin-bottom: 0px !important;
        }
    </style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "سلام امیدوارم خوب باشی. چطور میتونم کمکت کنم؟",
        }
    ]
if "model" not in st.session_state:
    st.session_state.model = genai.GenerativeModel("gemini-1.5-flash",
                              generation_config=generation_config,
                              system_instruction=system_instruction)
    
if "chat" not in st.session_state:
    st.session_state.chat = st.session_state.model.start_chat(history=[])

if "vision_encoder" not in st.session_state:
    st.session_state.vision_encoder = CLIPVisionModel.from_pretrained(IMAGE_EMBEDDING_MODEL)

if "preprocessor" not in st.session_state:
    st.session_state.preprocessor = CLIPFeatureExtractor.from_pretrained(IMAGE_EMBEDDING_MODEL)

if "text_encoder" not in st.session_state:
    st.session_state.text_encoder = RobertaModel.from_pretrained(TEXT_EMBEDDING_MODEL)

if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = AutoTokenizer.from_pretrained(TEXT_EMBEDDING_MODEL)

if "client" not in st.session_state:
    # client = weaviate.Client("http://localhost:8080")
    st.session_state.client = weaviate.connect_to_local()

if "result" not in st.session_state:
    st.session_state.result = None

if "option_image" not in st.session_state:
    st.session_state.option_image = False

def clear_chat_history():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "سلام امیدوارم خوب باشی. میتونی از کالای دلخواهت سوال کنی",
        }
    ]
    st.session_state.result = None

def write_prompt(prompt):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

def generate_llama2_response(prompt_input):
    string_dialogue = '''
    تو یک ربات فروشنده محصولات دیجیکالا هستی و تو وظیقه داری که با اطلاعاتی درباره محصولات جستجو شده در اختیارت قرار میدهم افراد رو راهنمایی کنی. به سوالات افراد درباره کالا پاسخ بدهی
    همیشه سعی کن پاسخ مفید بدهی و درباره هر آنچه که از تو سوال شده نظر بدهی. 
    ابتدا رسمی و ربات گونه و کوتاه خودت رو معرفی کن
    '''
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    user_input = f"{string_dialogue} {prompt_input} Assistant: "
    response = st.session_state.chat.send_message(user_input, generation_config=generation_config)
    return response.text

def llm_answer(prompt):
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if st.session_state.result:
                    text = ""
                    for obj in st.session_state.result.objects:
                        text += f"""
                        PID: {obj.properties['pid']},
                        "Title: {obj.properties['title']},
                        Description: {obj.properties['description']}
                        --------------
                        """
                    prompt = f"{prompt} اطلاعات کالا: {text}"
                else:
                    prompt = f"{prompt} کاربر کالایی انتخاب نکرده. بهش پیشنهاد بده از طریق ستون جستجوی کالا ویژگی کالا مورد نیازش رو توصیف کنه یا یک عکس مشابه به کالا آپلود کنه."
                response = generate_llama2_response(prompt)
                placeholder = st.empty()
                full_response = ""
                for item in response:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)

def llama_write_answer(prompt):
    write_prompt(prompt)
    llm_answer(prompt)

def get_text_vector(text):
    with torch.no_grad():
        output = st.session_state.text_encoder(**st.session_state.tokenizer(text, return_tensors='pt', truncation=True, padding=True))
    return output.pooler_output.squeeze(0).tolist()

def get_image_vector(image_path):
    image = Image.open(image_path).convert("RGB")
    with torch.no_grad():
        inputs = st.session_state.preprocessor(image, return_tensors='pt')
        output = st.session_state.vision_encoder(**inputs)
        out = output.pooler_output.squeeze(0)
        return out.tolist() 
    
def search_text_text(text):
    return st.session_state.client.collections.get("Product").query.near_vector(get_text_vector(text), target_vector="text_vector", limit=10)

def search_text_image(text):
    return st.session_state.client.collections.get("Product").query.near_vector(get_text_vector(text), target_vector="image_vector", limit=10)

def search_image_text(image):
    return st.session_state.client.collections.get("Product").query.near_vector(get_image_vector(image), target_vector="text_vector", limit=10)

def search_image_image(image):
    return st.session_state.client.collections.get("Product").query.near_vector(get_image_vector(image), target_vector="text_vector", limit=10)

def print_result():
    for obj in st.session_state.result.objects:
        print(f"PID: {obj.properties['pid']}")
        print(f"Title: {obj.properties['title']}")
        print(f"Description: {obj.properties['description']}")
        print("-" * 40)

def show_result():
    for obj in st.session_state.result.objects:
        col1, col2 = st.columns([1, 4])
        
        with col1:
            st.image(f'./product/images/{obj.properties['pid']}/{obj.properties['pid']}_0.jpg', width=60)  # Display small image icon

        with col2:
            st.markdown(f"**PID:** {obj.properties['pid']}")
            st.markdown(f"**Title:** [{obj.properties['title']}]({obj.properties['url']})")
            st.markdown(f"**Description:** {obj.properties['description']}")


with st.sidebar:
    st.markdown("# جستجوی کالا")
    st.divider()
    st.markdown("### تنظیم مدل زبانی")
    temperature = st.sidebar.slider(
        "temperature", min_value=0.0, max_value=2.0, value=0.1, step=0.01
    )
    top_p = st.sidebar.slider(
        "top_p", min_value=0.01, max_value=1.0, value=0.9, step=0.01
    )
    max_length = st.sidebar.slider(
        "max_length", min_value=32, max_value=128, value=120, step=8
    )
    st.button("پاک کردن تاریخچه", on_click=clear_chat_history)
    st.divider()
    st.markdown("### مدیریت پایگاه داده برداری")
    st.session_state.option_image = st.checkbox("جستجو در تصاویر؟")
    if st.button("اتصال"):
        if "client" not in st.session_state:
            st.session_state.client = weaviate.connect_to_local()
            st.success("Connected to Weaviate.")
    
    if st.button("قطع اتصال"):
        if "client" in st.session_state:
            st.session_state.client.close()
            del st.session_state.client
            st.success("Disconnected from Weaviate.")
st.title("ربات فروشنده دیجیکالا")
st.markdown("######")
st.divider()


i, col_1, j, col_2 = st.columns([1,10,1,15])

white_box_style = """
    <div style="background-color: green; height: 100px; width: 100%; border: 0px solid #ddd;"></div>
"""
with i:
    st.markdown(white_box_style, unsafe_allow_html=True)
with j:
    st.markdown(white_box_style, unsafe_allow_html=True)  

with col_1:
    st.markdown("## جستجوی کالا")
    text_query = st.text_input("کالای دلخواهت وارد")
    image_query = st.file_uploader("یک تصویر انتخاب کن", type=["jpg", "jpeg", "png"])


    if text_query:
        st.markdown(f'{text_query}')
        if st.session_state.option_image:
            st.session_state.result = search_text_image(text_query)
            print_result()
            show_result()
        else:
            st.session_state.result = search_text_text(text_query)
            print_result()
            show_result()

    if image_query:
        image = Image.open(image_query)
        st.image(image, caption="تصویر آپلود شده", width=300)
        if st.session_state.option_image:
            st.session_state.result = search_image_image(image_query)
            print_result()
            show_result()
        else:
            st.session_state.result = search_image_text(image_query)
            print_result()
            show_result()

with col_2:
    st.markdown("## چت با فروشنده")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    if prompt := st.chat_input(
        placeholder="یک سوال بپرس", max_chars=300
    ):
        write_prompt(prompt)

    llm_answer(prompt)
