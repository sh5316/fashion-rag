# ============================================================
# app_chat.py - FashionRAG '카톡 같은 채팅 화면' 버전 (홀 직원 2호)
# 켜는 방법:  streamlit run app_chat.py
# ------------------------------------------------------------
# app.py 와 하는 일은 똑같아요. '주방'(fashion_rag_core.py)에 요리를 부탁하고,
# 결과(사진 2장 + 조언)를 손님에게 보여 줘요.
# 다른 점은 '보여 주는 방법' 하나예요.
#   - app.py        : 입력 칸(편지 봉투) + 결과를 위에서 아래로 쌓아 보여 주기
#   - app_chat.py   : 카톡처럼 말풍선(st.chat_message) 으로 주고받기!
# ============================================================
import streamlit as st              # 화면 만드는 도구
import fashion_rag_core as core     # 진짜 요리는 '주방' 파일에 부탁해요

# set_page_config 는 꼭 '제일 먼저' 써야 하는 st 명령이에요. (브라우저 탭 이름 정하기)
st.set_page_config(page_title="FashionRAG 채팅", layout="centered")
st.title("👗 FashionRAG 채팅")
st.caption("패션·스타일링을 카톡처럼 물어보세요!")


# ------------------------------------------------------------
# 냉장고(캐시): Streamlit 은 말풍선을 보낼 때마다 이 파일을 1줄부터 다시 읽어요.
#   무거운 CLIP·DB·AI 줄은 한 번만 만들어 냉장고에 넣어 두고 꺼내 써요.
# ------------------------------------------------------------
@st.cache_resource(show_spinner="CLIP 모델과 벡터 DB를 불러오는 중...")
def load_collection():
    return core.get_collection()


@st.cache_resource(show_spinner=False)
def load_chains(vision_model):
    # 냉장고 칸은 '재료(vision_model)'별로 따로 있어요. 같은 모델이면 냉장고에서 꺼내요.
    return core.build_chains(vision_model=vision_model)


# ------------------------------------------------------------
# 0) 비밀 열쇠 확인 - 없으면 무서운 빨간 글씨 대신 '안내 문장'만 보여 주고 멈추기
# ------------------------------------------------------------
try:                               # try = "한번 해 보고"
    core.check_api_key()
except ValueError as e:            # except = "문제가 생기면 여기로"
    st.error(str(e))               # 빨간 안내 상자에 이유 보여 주기
    st.stop()                      # 여기서 멈춰! 아래 줄은 실행하지 않아요


# ------------------------------------------------------------
# 1) 옆 서랍(사이드바): 설정
# ------------------------------------------------------------
with st.sidebar:
    st.header("설정")
    vision_model = st.selectbox("비전 모델", ["gpt-4o-mini", "gpt-4o"],
                                help="gpt-4o 는 더 정확하지만 비용이 더 듭니다.")
    if st.button("대화 기록 지우기"):
        st.session_state.messages = []          # 말풍선 기록을 빈 목록으로 비우기


# ------------------------------------------------------------
# 2) 재료 준비 + 창고 정리 (손님 한 명당 처음 한 번만)
# ------------------------------------------------------------
collection = load_collection()                       # 냉장고에서 사진 창고 꺼내기

if not st.session_state.get("data_ready"):          # 사물함에 "준비 끝" 표시가 없으면
    with st.status("데이터 준비 상태 확인 중...", expanded=True) as status:
        bar = st.progress(0.0, text="이미지 확인 중")
        core.ensure_dataset(progress=lambda d, t: bar.progress(d / t, text=f"이미지 저장 {d}/{t}"))
        core.index_images(collection, progress=lambda d, t: bar.progress(d / t, text=f"임베딩 색인 {d}/{t}"))
        bar.progress(1.0, text="완료")
        status.update(label=f"준비 완료 - DB 이미지 {collection.count()}장", state="complete", expanded=False)
    st.session_state.data_ready = True               # 사물함에 "준비 끝!" 표시

st.sidebar.caption(f"DB 이미지 수: {collection.count()}장")
chains = load_chains(vision_model)                   # 고른 모델의 AI 줄을 냉장고에서 꺼내기


# ------------------------------------------------------------
# 3) 채팅 기록장: 말풍선들을 사물함(session_state) 에 차곡차곡 모아 둬요
#    한 칸(말풍선) 모양:
#       {"role": "user",      "content": "질문 글자"}
#       {"role": "assistant", "result":  {주방이 돌려준 결과 상자}}
#    결과 상자를 통째로 넣어 두면, 화면을 다시 그릴 때 사진도 그대로 다시 보여 줄 수 있어요.
# ------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


def render_answer(result):
    """AI 말풍선 하나를 그려 줘요 (닮은 사진들 → 스타일링 조언)."""
    st.caption(f"🔎 번역된 질문: {result['query_en']}")
    cols = st.columns(len(result["uris"]))            # 사진 수만큼 가로로 나란히 나누기
    for col, pid, uri, dist in zip(cols, result["ids"], result["uris"], result["distances"]):
        col.image(uri, caption=f"{pid} (거리 {dist:.3f})", use_container_width=True)
    st.markdown(result["answer_ko"])                  # AI 조언을 꾸민 글(마크다운)로 보여 주기


# ------------------------------------------------------------
# 4) 지금까지 쌓인 말풍선 전부 다시 그리기
#    (다시 읽기 때문에 매번 처음부터 그려야, 예전 대화가 사라지지 않아요)
# ------------------------------------------------------------
if not st.session_state.messages:
    with st.chat_message("assistant"):               # 첫 인사 말풍선
        st.markdown("안녕하세요! 어떤 스타일링이 궁금하신가요? 🙂\n\n"
                    "예) *하객룩으로 어울리는 원피스 코디*, *캐주얼한 데님 재킷 코디*")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):                # "user" / "assistant" 에 따라 좌우·아이콘이 달라져요
        if msg["role"] == "user":
            st.markdown(msg["content"])               # 손님 말풍선: 질문 글자
        else:
            render_answer(msg["result"])              # AI 말풍선: 사진 + 조언


# ------------------------------------------------------------
# 5) 맨 아래 입력 칸: st.chat_input
#    글자를 치고 엔터(또는 보내기)를 누르면 그 글자가 prompt 에 들어오고,
#    (:= 는 "값을 넣으면서 동시에 조건으로 확인하기" 라는 짧은 문법이에요)
#    아래 if 블록이 실행돼요. 비었으면 prompt 가 None 이라 아무 일도 안 해요.
# ------------------------------------------------------------
if prompt := st.chat_input("스타일링 질문을 입력하세요..."):
    # ① 손님 말풍선을 바로 보여 주고, 기록장에도 저장
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ② AI 말풍선: 주방에 요리를 부탁해서 결과를 보여 주기
    with st.chat_message("assistant"):
        try:
            with st.spinner("번역 → 이미지 검색 → 스타일링 조언 생성 중..."):
                result = core.ask(collection, chains, prompt)   # 주방에 요리 부탁!
            render_answer(result)
            # 결과 상자를 통째로 기록장에 저장 (다음 다시 읽기 때도 그대로 보이게)
            st.session_state.messages.append({"role": "assistant", "result": result})
        except Exception as e:
            # 인터넷이 끊기거나 AI 쪽 문제가 생겨도 앱 전체가 깨지지 않게, 안내 문장만 보여 줘요
            st.error(f"처리 중 오류가 발생했습니다: {e}")
