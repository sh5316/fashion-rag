# ============================================================
# fashion_rag_core.py  -  FashionRAG 의 '주방' (진짜 일을 하는 곳)
# ============================================================
# 식당에 비유하면 이 파일은 '주방'이에요.
#   - 사진 준비하기, 비슷한 사진 찾기, AI 에게 물어보기 같은 '요리'만 해요.
#   - 손님에게 보여 주는 '화면'은 여기서 만들지 않아요. (화면은 app.py = 홀 직원 담당)
# 주방을 따로 두면 좋은 점
#   ① 노트북에서 요리만 따로 맛보기(테스트) 쉬워요.
#   ② 나중에 홀 직원(화면)을 바꿔도 요리법은 그대로 다시 쓸 수 있어요.

import os        # 폴더와 파일을 다루는 도구
import base64    # 사진을 '글자'로 바꾸는 도구

from dotenv import load_dotenv                                        # .env 파일(비밀 열쇠)을 읽는 도구
import chromadb                                                       # 사진 창고(벡터 DB)
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction   # CLIP: 글·사진을 숫자로 바꾸는 AI
from chromadb.utils.data_loaders import ImageLoader                   # 파일 경로를 보고 사진을 열어 주는 도우미
from datasets import load_dataset                                     # 인터넷에서 사진 묶음 가져오기
from langchain_openai import ChatOpenAI                               # ChatGPT 같은 OpenAI 모델과 연결
from langchain_core.prompts import ChatPromptTemplate                 # AI 에게 보낼 '편지 양식'
from langchain_core.output_parsers import StrOutputParser             # AI 답장에서 글자만 쏙 꺼내기

load_dotenv()   # .env 파일 속 비밀 열쇠(OPENAI_API_KEY)를 꺼내서 컴퓨터가 쓸 수 있게 등록해요

# ------------------------------------------------------------
# 설정값 - 요리에 쓰는 '기본 레시피 숫자'들을 한곳에 모아 두기
# ------------------------------------------------------------
# __file__ = "이 파일이 어디에 있는지" 알려 주는 특별한 이름
# (노트북 칸 안에는 없지만, .py 파일 안에서는 쓸 수 있어요)

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))       # 이 파일이 있는 폴더
DATASET_FOLDER  = os.path.join(BASE_DIR, "fashion_dataset")   # 사진을 저장할 폴더
VDB_PATH        = os.path.join(BASE_DIR, "img_vdb")           # 사진 창고(Chroma DB) 폴더
COLLECTION_NAME = "fashion_images"                             # 창고 안 '선반' 이름
NUM_IMAGES      = 500        # 사진 몇 장 쓸까? (원래는 1000장. 시간이 넉넉하면 늘려도 돼요)
N_RESULTS       = 2          # AI 에게 보여 줄 사진 수 (편지 양식에 사진 칸이 2개라서 2장)
TRANSLATION_MODEL = "gpt-4o-mini"   # 번역을 맡을 AI (예전 gpt-3.5-turbo 대신 더 새것)
VISION_MODEL      = "gpt-4o-mini"   # 사진을 보고 설명하는 AI (앱 옆 서랍에서 바꿀 수 있어요)
IMAGE_EXTS = (".jpg", ".jpeg", ".png")   # "이 끝말로 끝나는 파일만 사진으로 칠게요"


def check_api_key():
    """비밀 열쇠(API 키)가 있는지 검사해요. 없으면 ValueError(오류)를 내요.
    파일을 불러오자마자 오류를 내면 앱 화면에 무서운 빨간 글씨가 잔뜩 뜨니까,
    앱에서 이 함수를 불러서 '친절한 안내 문장'으로 보여 주려고 따로 만들었어요."""
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY 가 없습니다. 노트북과 같은 폴더의 .env 파일에 키를 넣어 주세요.")


# ------------------------------------------------------------
# 파일 도우미 함수들
# ------------------------------------------------------------
def list_image_files(folder=DATASET_FOLDER):
    """폴더 안 사진 파일 이름을 '가나다(번호) 순'으로 줄 세워서 돌려줘요.
    폴더가 아직 없으면 빈 목록 [] 을 돌려줘요."""
    if not os.path.isdir(folder):
        return []
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTS))


def guess_mime(path):
    """파일 끝말(.png / .jpg)을 보고 '이건 어떤 종류의 사진이야'라는 이름표(MIME)를 붙여요.
    AI 에게 사진을 보낼 때 이 이름표가 필요해요."""
    return "image/png" if path.lower().endswith(".png") else "image/jpeg"


def encode_image(path):
    """사진을 base64 '글자'로 바꿔요.
    AI 에게 보내는 편지는 글자만 들어갈 수 있어서, 사진을 글자로 바꿔서 넣어야 해요."""
    with open(path, "rb") as f:                       # rb = 사진을 있는 그대로(바이트로) 읽기
        return base64.b64encode(f.read()).decode("utf-8")


# ------------------------------------------------------------
# 1) 재료 준비 - 패션 사진 모으기
# ------------------------------------------------------------
def ensure_dataset(num_images=NUM_IMAGES, progress=None):
    """사진이 num_images 장보다 적으면 인터넷(Fashionpedia)에서 받아와 저장해요.

    progress : "중간중간 알려 줘!" 하고 맡겨 두는 '연락 함수(콜백)'예요. 모양: (done, total)
               노트북은 print(글자 출력)를, 웹 앱은 진행 막대 함수를 넘겨줘요.
               => 심부름꾼(이 함수)은 같고, 연락받는 방법만 바꿔 끼울 수 있어요!
    """
    os.makedirs(DATASET_FOLDER, exist_ok=True)       # 사진 폴더 만들기 (이미 있으면 그냥 넘어가요)
    if len(list_image_files()) >= num_images:
        return 0                                   # 사진이 이미 충분하면 아무것도 안 하고 끝!


    # 전체(약 3.5GB)를 다 받으면 너무 오래 걸려요.
    # streaming=True = 수도꼭지처럼 '필요한 만큼만' 조금씩 받아 오기

    dataset = load_dataset("detection-datasets/fashionpedia", split="train", streaming=True)
    saved = 0                                        # 지금까지 저장한 사진 수

    for i, example in enumerate(dataset.take(num_images)):   # 앞에서부터 num_images 장만 꺼내기

        # 파일 이름을 image_0001.jpg 처럼 '0을 채운 4자리'로 만들어요.
        # (image_1, image_10, image_2 처럼 순서가 뒤죽박죽 되는 걸 막아 줘요)

        path = os.path.join(DATASET_FOLDER, f"image_{i + 1:04d}.jpg")
        # 사진 색 방식을 RGB 로 통일해서 JPG 로 저장 (투명 배경 사진은 JPG 로 저장이 안 되거든요)

        example["image"].convert("RGB").save(path, format="JPEG", quality=90)
        saved += 1
        if progress:                                 # 연락 함수를 받았다면
            progress(saved, num_images)              # "지금 몇 장 중 몇 장 했어요!" 하고 알려 주기
    return saved


# ------------------------------------------------------------
# 2) 사진 창고(벡터 DB)
# ------------------------------------------------------------
def get_collection():
    """CLIP(글·사진을 숫자 지도로 바꾸는 AI)이 연결된 사진 창고 선반을 돌려줘요.
    CLIP 은 약 600MB 로 아주 무거워요. 그래서 앱에서는 '냉장고(캐시)'에 넣어 딱 한 번만 불러요."""

    client = chromadb.PersistentClient(path=VDB_PATH)      # 창고 열기 (껐다 켜도 내용이 남는 창고)
    return client.get_or_create_collection(               # 선반이 있으면 가져오고, 없으면 새로 만들기
        name=COLLECTION_NAME,
        embedding_function=OpenCLIPEmbeddingFunction(),   # 글이든 사진이든 '같은 지도' 위 좌표(숫자 512개)로 바꿔 줌
        data_loader=ImageLoader(),                        # 파일 주소만 주면 사진을 열어 주는 도우미
    )


def index_images(collection, batch_size=50, progress=None):
    """폴더 사진 중에서 창고에 '아직 없는 사진만' 골라 지도에 점을 찍어(등록해) 둬요.
        collection.get(include=[]) 로 '이름표(id) 목록만' 가볍게 받아 와요.
    """

    files = list_image_files()                                  # 폴더에 있는 사진들
    existing = set(collection.get(include=[])["ids"])           # 창고에 이미 있는 사진 이름표들

    # 이름표(id) = 파일 이름에서 .jpg 를 뗀 것 (예: image_0001)
    # 그래야 "창고의 image_0001 = 폴더의 image_0001.jpg" 로 딱 맞아떨어져요.

    todo = [(os.path.splitext(f)[0], os.path.join(DATASET_FOLDER, f))
            for f in files if os.path.splitext(f)[0] not in existing]   # 창고에 없는 것만 '할 일 목록'에
    for start in range(0, len(todo), batch_size):               # 50장씩 묶어서 차례대로
        batch = todo[start:start + batch_size]
        collection.add(ids=[i for i, _ in batch], uris=[u for _, u in batch])   # 이름표 + 파일 주소로 등록
        if progress:
            progress(min(start + batch_size, len(todo)), len(todo))             # "몇 장 했어요!" 연락
    return len(todo)                                            # 새로 등록한 사진 수


def query_db(collection, query, n_results=N_RESULTS):
    """글(query)과 지도에서 가장 가까운 = 가장 닮은 사진을 n_results 장 찾아요.
    결과는 [[...]] 처럼 상자 안의 상자 모양이에요. (바깥 상자 = 질문, 안쪽 상자 = 1등, 2등...)"""
    return collection.query(query_texts=[query], n_results=n_results,
                            include=["uris", "distances"])     # 사진 주소 + 거리(작을수록 닮음)도 같이 받기


# ------------------------------------------------------------
# 3) AI 연결 줄(체인) 만들기 - LangChain
# ------------------------------------------------------------
def build_chains(translation_model=TRANSLATION_MODEL, vision_model=VISION_MODEL):
    """두 개의 AI 연결 줄을 만들어서 (번역 체인, 비전 체인) 으로 돌려줘요.
    체인 = '편지 양식 | AI | 답장에서 글자만 꺼내기' 를 기차처럼 | 로 이어 붙인 것"""
    #    
    #  {target_lang} 빈칸을 두고, 체인은 한 번만 만들어 계속 써요.

    translation_chain = (
        ChatPromptTemplate.from_messages([
            # system = AI 에게 주는 '역할 카드': "너는 번역가야. {target_lang}(으)로 번역해 줘"
            ("system", "You are a professional translator. Translate the following text to {target_lang}. "
                       "Keep markdown formatting. Output only the translation."),
            ("user", "{text}"),                                  # 번역할 글이 들어갈 빈칸
        ])
        | ChatOpenAI(model=translation_model, temperature=0.0)  # temperature=0 = 매번 똑같이, 차분하게 대답
        | StrOutputParser()                                     # 답장에서 글자만 꺼내기
    )

    vision_chain = (
        ChatPromptTemplate.from_messages([
            # 역할 카드: "너는 친절한 패션 도우미야. 사진을 보면서 대화하듯 설명해 줘"
            ("system",
             "You are a helpful fashion and styling assistant. Answer the user's question using the given "
             "image context with direct references to parts of the images provided. Maintain a more "
             "conversational tone, don't make too many lists. Use markdown formatting for highlights, "
             "emphasis, and structure."),
            ("user", [
                {"type": "text", "text": "What are some ideas for styling {user_query}"},   # 질문 빈칸
                # 사진 칸 2개: data:이름표;base64,글자로-바꾼-사진 모양으로 넣어요
                {"type": "image_url", "image_url": {"url": "data:{image_mime_1};base64,{image_data_1}"}},
                {"type": "image_url", "image_url": {"url": "data:{image_mime_2};base64,{image_data_2}"}},
            ]),
        ])
        | ChatOpenAI(model=vision_model, temperature=0.0)
        | StrOutputParser()
    )
    return translation_chain, vision_chain


def format_prompt_inputs(results, user_query):
    """찾은 사진 중 1등·2등 사진과 질문을 '편지 양식의 빈칸'에 채울 수 있게 사전(dict)으로 만들어요."""
    uris = results["uris"][0]                     # 첫 번째 질문의 결과 사진 주소들
    if len(uris) < 2:   # 사진이 2장보다 적으면, 알아보기 쉬운 말로 오류를 알려 줘요
        raise ValueError(f"이미지가 최소 2장 필요합니다. (검색 결과 {len(uris)}장)")
    inputs = {"user_query": user_query}           # 질문 빈칸 채우기
    for n, path in enumerate(uris[:2], start=1):  # n = 1, 2
        inputs[f"image_data_{n}"] = encode_image(path)   # 사진 → 글자
        inputs[f"image_mime_{n}"] = guess_mime(path)     # 사진 종류 이름표
    return inputs


def ask(collection, chains, query_ko, n_results=N_RESULTS):
    """한국어 질문 하나를 받아서 → 번역 → 사진 찾기 → AI 조언 → 다시 한국어로!
    화면에 보여 줄 것들을 사전(dict) 한 상자에 담아서 돌려줘요."""
    translation_chain, vision_chain = chains
    query_en = translation_chain.invoke({"text": query_ko, "target_lang": "English"})   # ① 영어로 번역
    results = query_db(collection, query_en, n_results)                                # ② 닮은 사진 찾기 (R: 찾기)
    inputs = format_prompt_inputs(results, query_en)                                   # ③ 편지 빈칸 채우기 (A: 붙이기)
    answer_en = vision_chain.invoke(inputs)                                            # ④ AI 가 사진 보고 답 (G: 만들기)
    answer_ko = translation_chain.invoke({"text": answer_en, "target_lang": "Korean"}) # ⑤ 한국어로 번역
    return {
        "query_ko": query_ko, "query_en": query_en,                                     # 질문 (한국어/영어)
        "ids": results["ids"][0], "uris": results["uris"][0], "distances": results["distances"][0],  # 찾은 사진 정보
        "answer_ko": answer_ko,                                                         # 최종 답
    }
