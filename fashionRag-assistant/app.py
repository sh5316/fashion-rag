# ============================================================
# app.py - FashionRAG 의 '홀 직원' (손님에게 보여 주는 화면 담당)
# 켜는 방법:  streamlit run app.py
# ============================================================
import streamlit as st              # 화면 만드는 도구
import fashion_rag_core as core     # 진짜 요리는 '주방' 파일에 부탁해요

# set_page_config 는 꼭 '제일 먼저' 써야 하는 st 명령이에요.
# (브라우저 탭에 보일 이름 + 화면을 넓게 쓰기)
st.set_page_config(page_title="FashionRAG", layout="wide")
st.title("FashionRAG: 패션 및 스타일링 어시스턴트")


# ------------------------------------------------------------
# 냉장고(캐시): Streamlit 은 버튼을 누를 때마다 이 파일을 1줄부터 다시 읽어요.
#   무거운 것을 매번 새로 만들면 너무 느리니까, 한 번 만들어 냉장고에 넣어 두고 꺼내 써요.
# [원래 코드의 문제] 질문할 때마다 600MB 짜리 CLIP 을 다시 불러와서 엄청 느렸어요.
# ------------------------------------------------------------
@st.cache_resource(show_spinner="CLIP 모델과 벡터 DB를 불러오는 중...")   # 처음 만들 때 이 글자를 보여 줘요
def load_collection():
    return core.get_collection()


@st.cache_resource(show_spinner=False)
def load_chains(vision_model):
    # 냉장고 칸은 '재료(vision_model)'별로 따로 있어요.
    # 다른 모델을 고르면 새로 만들고, 전에 골랐던 모델이면 냉장고에서 꺼내요.
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
with st.sidebar:                   # 이 안에 쓴 것은 화면 왼쪽 서랍에 들어가요
    st.header("설정")
    vision_model = st.selectbox("비전 모델", ["gpt-4o-mini", "gpt-4o"],     # 골라 고르기 상자
                                help="gpt-4o 는 더 정확하지만 비용이 더 듭니다.")
    if st.button("대화 기록 지우기"):
        st.session_state.history = []                # 사물함 속 기록을 빈 목록으로 비우기


# ------------------------------------------------------------
# 2) 재료 준비 + 창고 정리 (손님 한 명당 처음 한 번만 확인)
#    session_state(사물함) 에 "준비 끝!" 표시를 해 두면 다시 실행돼도 또 하지 않아요
# ------------------------------------------------------------
collection = load_collection()                       # 냉장고에서 사진 창고 꺼내기

if not st.session_state.get("data_ready"):          # 사물함에 "준비 끝" 표시가 없으면
    with st.status("데이터 준비 상태 확인 중...", expanded=True) as status:   # 진행 상황 상자
        bar = st.progress(0.0, text="이미지 확인 중")                          # 진행 막대 (0%)
        # 심부름꾼에게 '진행 막대 채우기 함수'를 연락 방법(콜백)으로 맡겨요
        # lambda d, t: ... = 이름 없는 짧은 함수. d = 한 개수, t = 전체 개수
        core.ensure_dataset(progress=lambda d, t: bar.progress(d / t, text=f"이미지 저장 {d}/{t}"))
        core.index_images(collection, progress=lambda d, t: bar.progress(d / t, text=f"임베딩 색인 {d}/{t}"))
        bar.progress(1.0, text="완료")                                          # 막대 100%
        status.update(label=f"준비 완료 - DB 이미지 {collection.count()}장", state="complete", expanded=False)
    st.session_state.data_ready = True               # 사물함에 "준비 끝!" 표시

st.sidebar.caption(f"DB 이미지 수: {collection.count()}장")   # 서랍에 작은 글씨로 사진 수 보여 주기
chains = load_chains(vision_model)                   # 고른 모델의 AI 줄을 냉장고에서 꺼내기

if "history" not in st.session_state:
    st.session_state.history = []        # 사물함에 '질문 기록장' 만들기 (질문·사진·답이 하나씩 쌓여요)


# ------------------------------------------------------------
# 3) 질문 받기 - 편지 봉투(form): '조언 받기' 버튼을 눌렀을 때만 처리해요
# [원래 코드의 문제] 입력 칸에 글자가 남아 있어서, 다른 버튼만 눌러도 AI 를 또 불렀어요(돈이 들어요!)
# ------------------------------------------------------------
st.header("스타일링 조언을 받아보세요")
with st.form("ask_form", clear_on_submit=True):     # clear_on_submit = 제출하면 입력 칸을 깨끗이 비우기
    query_ko = st.text_input("스타일링에 대한 질문을 입력하세요:",
                             placeholder="예) 하객룩으로 어울리는 원피스 코디")   # 흐린 예시 글자
    submitted = st.form_submit_button("조언 받기")

if submitted and query_ko.strip():                   # 제출했고, 빈칸이 아닐 때만 (strip = 앞뒤 빈칸 지우기)
    try:
        with st.spinner("번역 -> 이미지 검색 -> 스타일링 조언 생성 중..."):   # 빙글빙글 기다림 표시
            result = core.ask(collection, chains, query_ko.strip())       # 주방에 요리 부탁!
        st.session_state.history.insert(0, result)   # 새 결과를 기록장 '맨 앞'에 끼워 넣기
    except Exception as e:
        # 인터넷이 끊기거나 AI 쪽에 문제가 생겨도 앱 전체가 깨지지 않게, 안내 문장만 보여 줘요
        st.error(f"처리 중 오류가 발생했습니다: {e}")


# ------------------------------------------------------------
# 4) 결과 보여 주기 - 가장 최근 것은 크게, 예전 것은 접어서
# ------------------------------------------------------------
def render(result):
    """결과 하나를 화면에 그려 주는 함수 (질문 영어 번역 → 사진들 → AI 조언)"""
    st.caption(f"번역된 질문: {result['query_en']}")
    cols = st.columns(len(result["uris"]))            # 사진 수만큼 화면을 가로로 나란히 나누기
    for col, pid, uri, dist in zip(cols, result["ids"], result["uris"], result["distances"]):
        # st.image 는 사진 파일 주소만 줘도 바로 보여 줘요 (글자로 바꿀 필요 없어요)
        col.image(uri, caption=f"{pid} (거리 {dist:.3f})", width=300)
    st.markdown(result["answer_ko"])                  # AI 조언을 꾸민 글(마크다운)로 보여 주기


if st.session_state.history:                         # 기록장에 뭔가 있으면
    latest, *older = st.session_state.history         # 맨 앞 1개 = latest(최신), 나머지 = older(예전)
    st.subheader(f"Q. {latest['query_ko']}")
    render(latest)
    if older:
        st.divider()                                  # 가로줄 긋기
        st.subheader("이전 질문")
        for past in older:
            with st.expander(f"Q. {past['query_ko']}"):   # 접었다 폈다 하는 상자
                render(past)
else:
    st.info("질문을 입력하면 비슷한 패션 사진 2장과 함께 스타일링 조언을 드립니다.")   # 아직 질문이 없을 때 파란 안내
