from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline, DonutProcessor, AutoTokenizer, AutoModelForImageTextToText
from PIL import Image
import numpy as np
import cv2
import pytesseract
import io
import re
import uvicorn
import requests
import httpx
import json
from paddleocr import PaddleOCR
from openai import OpenAI
import os
from dotenv import load_dotenv
from datasets import load_dataset
# from langchain_community.vectorstores import Chroma
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.schema import Document

# 한국어 임베딩 모델
# embedding_model = HuggingFaceEmbeddings(
#     model_name="jhgan/ko-sroberta-multitask"
# )

# Chroma 생성
# vectorstore = Chroma(
#     collection_name="store_info",
#     embedding_function=embedding_model,
#     persist_directory="./chroma_db"
# )

# sample data
# store_data = [
#     {"store": "메저커피", "address": "대전광역시 유성구 신성로83번길 15", "대표": "김승연", "사업자등록번호": "7852501821"},
#     {"store": "스타벅스 대전시청점", "address": "대전 서구 둔산동 1234", "대표": "이카루스", "사업자등록번호": "1029384756"},
# ]

# vector store에 추가
# docs = []
# for store in store_data:
#     content = f"상호명: {store['store']}, 주소: {store['address']}, 대표명: {store['대표']}, 사업자등록번호: {store['사업자등록번호']}"
#     docs.append(Document(page_content=content))

# if vectorstore._collection.count() == 0:
#     vectorstore.add_documents(docs)
#     vectorstore.persist()
#     print("✅ VectorStore에 문서 추가 완료")
# else:
#     print("✅ 기존 VectorStore 사용")


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

ds = load_dataset("naver-clova-ix/cord-v2")
sample = ds["train"][0]
# print("sample", sample)

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
# model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ocr/donut-base-finetuned-cord-v1")
# tokenizer = AutoTokenizer.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = AutoModelForImageTextToText.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")

ocr_engine = PaddleOCR(lang="korean", use_angle_cls=True)

# vocab = processor.tokenizer.get_vocab()
# for token in vocab:
#     if token.startswith("<s_"):
#         print(token)

# token 정보
# <s_price>
# <s_othersvc_price>
# <s_itemsubtotal>
# <s_cnt>
# <s_num>
# <s_void_menu>
# <s_sub_total>
# <s_emoneyprice>
# <s_discountprice>
# <s_tax_price>
# <s_sub>
# <s_cashprice>
# <s_total_etc>
# <s_unitprice>
# <s_creditcardprice>
# <s_total_price>
# <s_total>
# <s_menu>
# <s_menuqty_cnt>
# <s_nm>
# <s_changeprice>
# <s_etc>
# <s_service_price>
# <s_vatyn>
# <s_synthdog>
# <s_subtotal_price>
# <s_discount_price>
# <s_menutype_cnt>
# <s_iitcdip>
# <s_cord-v2>


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 보안을 위해 나중에 도메인 제한 가능
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class OCRText(BaseModel):
    text: str

# 모델 경로
MODEL_PATH = {
    # "custom_bert": "내_튜닝된_BERT_모델_폴더",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    # "mistral-8x-7B": "mistralai/Mixtral-8x7B-Instruct-v0.1", ==> model 로드하는 중 죽어버림.
    "mistral-quantized": "TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ",
    "phi3-4K": "microsoft/Phi-3-mini-4k-instruct",
    "phi3-128K": "microsoft/Phi-3-mini-128k-instruct"
}

# NER 파이프라인
# ner_pipeline = pipeline(
#     "ner",
#     model="klue/ner",
#     tokenizer="klue/ner",
#     aggregation_strategy="simple"
# )

# LLM 파이프라인
# llm_pipeline = pipeline(
#     "text-generation",
#     model=MODEL_PATH["mistral-quantized"],  # 사용 시 Phi-3로 바꿈
#     tokenizer=MODEL_PATH["mistral-quantized"],
#     max_new_tokens=64,
#     device_map="auto"
# )

def extract_json(text):
    try:
        start = text.find('{')
        if start == -1:
            return None
        count = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                count += 1
            elif text[i] == '}':
                count -= 1
                if count == 0:
                    json_text = text[start:i+1]
                    return json.loads(json_text)
        return None
    except Exception as e:
        print("JSON 파싱 에러:", e)
        return None  # 파싱 실패하면 None 반환

ollama_host = "http://localhost:11434"

# ollama 파이프라인
async def query_ollama_phi3(prompt: str) -> str:
    async with httpx.AsyncClient(timeout=600.0) as client:
        response = await client.post(
            f"{ollama_host}/api/generate",
            json={
                "model": "phi3:mini",
                "prompt": prompt,
                "stream": False,
                "num_predict": 1024
            }
        )
        result = response.json()
        if "response" in result:
            return result["response"]
        else:
            print("=== Ollama 에러 ===", result)
            raise ValueError(f"Ollama 응답에 'response'가 없습니다. 전체 결과: {result}")

# GPT-4O-MINI 파이프라인
def query_gpt4o_mini(prompt: str) -> str:
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "당신은 OCR된 영수증을 분석하는 전문가입니다."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )
    print("=== GPT-4O-MINI 결과 ===", response.id)
    print("=== GPT-4O-MINI 토큰 사용량 ===", response.usage)
    print("=== GPT-4O-MINI api_key ===", api_key)
    return response.choices[0].message.content

def ocr_with_paddle(image: Image.Image) -> str:
    img_np = np.array(image.convert("RGB"))
    result = ocr_engine.ocr(img_np)
    lines = []
    for line in result[0]:
        lines.append(line[1][0])
    return "\n".join(lines)

@app.post("/ocr_paddle")
async def ocr_paddle(file: UploadFile):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    text = ocr_with_paddle(image)

    max_lines = 20  # 최대 20줄만 사용
    lines = text.split("\n")
    shortened_text = "\n".join(lines[:max_lines])
    
    prompt = f"""
    다음은 영수증의 OCR 텍스트입니다.

    다음 항목들을 JSON 형태로 정확하게 추출하세요.
    항목 설명:
    - "store" : 상호명 (예: 메저커피)
    - "representative" : 대표명 (예: 김승연)
    - "business_number" : 사업자등록번호 (예: 782501821)
    - "address" : 주소 (예: 대전광역시 유성구 신성로83번길 15 (신성동) 102호)
    - "card_number" : 카드번호 (예: 54287967****478*)
    - "payment_method" : 결제방법 (예: 신용카드, 체크카드, 현금 등)
    - "approval_number" : 승인번호 (8자리 숫자, 예: 45895407)
    - "approval_datetime" : 승인일시 (YYYY-MM-DD HH:MM:SS 형식, 예: 2025-03-28 10:31:21)
    - "items" : 구매 항목 리스트 (메뉴명, 단가, 수량, 총액)

    반드시 아래 JSON 형식으로 출력하세요:

    {{
        "store": "",
        "representative": "",
        "business_number": "",
        "address": "",
        "card_number": "",
        "payment_method": "",
        "approval_number": "",
        "approval_datetime": "",
        "items": [
            {{"menu_name": "", "unit_price": "", "quantity": "", "amount": ""}}
        ]
    }}

    영수증 텍스트:
    {shortened_text}
    """

    llm_response = await query_ollama_phi3(prompt)
    parsed_json = extract_json(llm_response)

    if parsed_json is None:
        llm_response = query_gpt4o_mini(prompt)

    # store_name = parsed_json.get("store", "") if parsed_json else ""
    # address = parsed_json.get("address", "")
    # rag_results = search_store_info(store_name, address)

    return {
        "ocr_result": text,
        "llm_result": parsed_json,
        # "rag_result": rag_results
    }

# def search_store_info(store_name: str, address: str = ""):
#     query = f"{store_name} {address}"
#     results = vectorstore.similarity_search(query, k=3)  # 유사 top3
#     if results:
#         return [doc.page_content for doc in results]
#     else:
#         return ["관련 매장 정보를 찾을 수 없습니다."]

# 이미지 전처리 함수
def preprocess_image_for_ocr(image: Image.Image) -> Image.Image:
    # PIL 이미지를 OpenCV로 변환
    img_cv = np.array(image)
    if len(img_cv.shape) == 3:
        img_gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_cv

    # --- 1. 기울기 보정 (deskew) ---
    coords = np.column_stack(np.where(img_gray > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = img_gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    img_rotated = cv2.warpAffine(img_gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # --- 2. Adaptive Thresholding ---
    img_thresh = cv2.adaptiveThreshold(
        img_rotated, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 15
    )

    # --- 3. 노이즈 제거 ---
    img_blurred = cv2.medianBlur(img_thresh, 3)

    # 다시 PIL 이미지로 변환
    preprocessed_pil = Image.fromarray(img_blurred)
    return preprocessed_pil


# OCR 전처리 함수
def clean_ocr_text(text: str) -> str:
    # 1. 불필요 문자 제거
    text = re.sub(r'[^\w\s\d가-힣.,():-]', ' ', text)
    text = re.sub(r'\.{2,}', ',', text)
    text = re.sub(r'\s+', ' ', text)

    # 2. 가격 쉼표 제거 (0,200 → 0200 방지)
    text = re.sub(r'(\d),(\d)', r'\1\2', text)

    # 3. 금액 라인만 남기거나, 날짜 추출 시도 가능 (여기선 일단 일반 정리)
    return text.strip()

# LLM 테스트 함수
# @app.get("/llm_test")
# def llm_test():
#     prompt = "안녕하세요. 당신은 영수증 분석 AI입니다. 간단히 자기소개해 주세요."
#     llm_response = llm_pipeline(prompt)[0]["generated_text"]
#     return {"llm_result": llm_response}

# OCR 결과 출력하는 함수
@app.post("/ocr_tesseract")
async def ocr_tesseract(file: UploadFile):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # 이미지 전처리
    image = preprocess_image_for_ocr(image)

    # OCR
    text = pytesseract.image_to_string(image, lang='kor')
    return {"ocr_result": text}

class OCRText(BaseModel):
    text: str

def parse_receipt_with_donut(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((960, 960))

    task_prompt = "<s_cord-v2>"
    inputs = processor(images=image, text=task_prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=2048)
    result = processor.batch_decode(outputs, skip_special_tokens=True)[0]

    # 태그 기반 파싱
    # parsed_result = parse_donut_output(result)
    return result

def parse_donut_output(result: str):
    fields = [
        "price", "othersvc_price", "itemsubtotal", "cnt", "num", "void_menu",
        "sub_total", "emoneyprice", "discountprice", "tax_price", "sub",
        "cashprice", "total_etc", "unitprice", "creditcardprice", "total_price",
        "total", "menu", "menuqty_cnt", "nm", "changeprice", "etc",
        "service_price", "vatyn", "synthdog", "subtotal_price", "discount_price",
        "menutype_cnt", "iitcdip"
    ]

    result_dict = {}
    for field in fields:
        pattern = f"<s_{field}>(.*?)</s_{field}>"
        matches = re.findall(pattern, result)
        if matches:
            result_dict[field] = matches
        else:
            result_dict[field] = []

    return result_dict

    # # 품목명
    # names = re.findall(r"<s_nm>\s*(.*?)\s*</s_nm>", donut_text)
    # prices = re.findall(r"<s_price>\s*(.*?)\s*</s_price>", donut_text)
    # # 수량/가격이 num으로 나오는 경우도 고려
    # nums = re.findall(r"<s_num>\s*(.*?)\s*</s_num>", donut_text)

    # # 항목 구성
    # for i in range(max(len(names), len(prices), len(nums))):
    #     item = {
    #         "name": names[i] if i < len(names) else None,
    #         "price": None
    #     }
    #     # price 우선 순위: s_price -> s_num
    #     if i < len(prices) and prices[i].strip():
    #         item["price"] = prices[i].strip()
    #     elif i < len(nums) and nums[i].strip():
    #         item["price"] = nums[i].strip()
    #     result["items"].append(item)

    # # 상호명 찾기 (예: store 태그가 있다면)
    # store_match = re.search(r"<s_store>\s*(.*?)\s*</s_store>", donut_text)
    # if store_match:
    #     result["store"] = store_match.group(1).strip()

    # # 날짜 찾기 (예: s_date 태그 예상)
    # date_match = re.search(r"<s_date>\s*(.*?)\s*</s_date>", donut_text)
    # if date_match:
    #     result["date"] = date_match.group(1).strip()

    # return result

def strip_tags(text: str) -> str:
    result = ""
    # 각 태그에 따라 라벨링 추가
    tag_map = {
        "nm": "메뉴명",
        "price": "가격",
        "num": "숫자",
        "unitprice": "단가",
        "cnt": "수량",
        "creditcardprice": "신용카드금액",
        "total_price": "총금액",
        "approval_number": "승인번호",
        "approval_datetime": "승인일시",
        "etc": "기타",
        "service_price": "서비스가격",
        "tax_price": "부가세",
        "changeprice": "거스름돈",
    }
    for field, label in tag_map.items():
        text = re.sub(
            fr"<s_{field}>(.*?)</s_{field}>",
            lambda m: f"{label}: {m.group(1).strip()} | ",
            text
        )
    # 태그 없는 나머지는 제거
    text = re.sub(r"</?s_[^>]+>", "", text)
    text = re.sub(r"<sep\s*/?>", " | ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

@app.post("/ocr_donut")
async def ocr_donut(file: UploadFile):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    # image = preprocess_image_for_ocr(image)
    donut_result_text = parse_receipt_with_donut(image)  # 이미 dict
    # return {"ocr_result": parsed_result}
    clean_text = strip_tags(donut_result_text)

    prompt = f"""
    다음은 OCR 및 Donut으로 추출한 영수증 텍스트입니다.
    상호명(store), 대표명(representative), 사업자등록번호(business_number), 주소(address),
    카드번호(card_number), 결제방법(payment_method), 승인번호(approval_number), 승인일시(approval_datetime),
    메뉴명(menu_name), 단가(unit_price), 수량(quantity), 금액(amount) 필드를 JSON으로 출력하세요.

    설명 없이 아래 형식만 출력하세요:
    {{
        "store": "",
        "representative": "",
        "business_number": "",
        "address": "",
        "card_number": "",
        "payment_method": "",
        "approval_number": "",
        "approval_datetime": "",
        "items": [{{"menu_name": "", "unit_price": "", "quantity": "", "amount": ""}}]
    }}

    영수증 텍스트:
    {clean_text}
    """

    llm_response = query_ollama_phi3(prompt)
    parsed_json = extract_json(llm_response)

    if parsed_json is None:
        parsed_json = llm_response

    return {
        "donut_result": donut_result_text,
        "plain_text": clean_text,
        "llm_result": parsed_json
    }

# 영수증 파싱함수
@app.post("/parse_receipt")
async def parse_receipt(data: OCRText):
    print("=== parse_receipt 시작 ===")
    raw_text = data.text
    cleaned_text = clean_ocr_text(raw_text)

    # print("=== OCR 원본 ===")
    # print(raw_text)
    # print("=== 전처리 후 ===")
    # print(cleaned_text)

    # ner_entities = ner_pipeline(cleaned_text)
    # print("=== NER 결과 ===")
    # print(ner_entities)

    # ner_result = []
    # for entity in ner_entities:
    #     ner_result.append({
    #         "entity": entity["entity_group"],
    #         "word": entity["word"]
    #     })
    
    # NER에서 상호명/날짜/금액을 미리 추출
    # store_names = [e["word"] for e in ner_result if e["entity"] == "ORG"]
    # dates = [e["word"] for e in ner_result if e["entity"] == "DATE"]
    # prices = [e["word"] for e in ner_result if e["entity"] in ["QUANTITY", "MONEY"]]

    # 가장 가능성 높은 값만 사용
    # store = store_names[0] if store_names else "알수없음"
    # date = dates[0] if dates else "알수없음"

    prompt = f"""
    다음은 OCR로 추출한 영수증입니다.
    항목명(name), 가격(price), 상호명(store), 날짜(date)를 JSON으로 추출하세요.
    설명 없이 아래 형식만 출력하세요:
    {{
    "items": [{{"name": "품목", "price": "금액"}}],
    "store": "상호명",
    "date": "YYYY-MM-DD"
    }}

    영수증:
    {cleaned_text}
    """

    print("=== llm_pipeline 호출 시작 ===") 
    # llm_response = llm_pipeline(prompt)[0]["generated_text"]
    llm_response = await query_ollama_phi3(prompt)
    parsed_json = extract_json(llm_response)

    if parsed_json is None:
        parsed_json = llm_response
    print("=== llm_pipeline 호출 끝 ===")

    # if api_key:
    #     print("=== gpt4o_mini 호출 시작 ===")
    #     llm_response_gpt4o = query_gpt4o_mini(prompt)
    #     print("=== gpt4o_mini 호출 끝 ===")
    # else:
    #     llm_response_gpt4o = "api_key 없음"

    print("=== parse_receipt 끝 ===")
    return {
        "ner_result": [],
        "llm_result": parsed_json,
        "llm_result_gpt4o": [],
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)