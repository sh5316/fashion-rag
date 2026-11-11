# ============================================================
# hello_app.py - Streamlit 이 어떻게 움직이는지 체험하는 꼬마 앱
# ============================================================
import time                 # 시계 도구
import streamlit as st      # Streamlit 을 'st' 라는 짧은 별명으로 불러오기

st.title("Streamlit 동작 원리 체험")              # 화면 맨 위에 큰 제목 쓰기

# ── ① 다시 실행(rerun) 확인 ──────────────────────────────────
# 버튼 같은 걸 누를 때마다 이 파일 전체가 '1줄부터 다시' 실행돼요.
# 그래서 버튼을 누를 때마다 아래 시각이 새로 바뀌어요!
st.write("이 스크립트가 실행된 시각:", time.strftime("%H:%M:%S"))

# ── ② 칠판(일반 변수) vs 사물함(session_state) ────────────────
count = 0                                          # 칠판: 다시 실행될 때마다 0 으로 지워져요
if "count" not in st.session_state:                # 사물함에 count 가 아직 없다면 (맨 처음 한 번만)
    st.session_state.count = 0                     #   사물함에 0 을 넣어 두기

if st.button("+1 누르기"):                         # 버튼을 '누른 바로 그 순간'의 실행에서만 True(참)
    count += 1    #                                 # 칠판 숫자 +1 (다음 실행엔 다시 0)
    st.session_state.count += 1                    # 사물함 숫자 +1 (계속 남아요)

st.write("일반 변수 count =", count)               # 항상 0 아니면 1
st.write("session_state.count =", st.session_state.count)   # 누를수록 1, 2, 3... 계속 커져요

# ── ③ 냉장고(캐시) ────────────────────────────────────────────
@st.cache_resource                                  # "이 함수의 결과는 냉장고에 넣어 두고 다시 꺼내 써!"
def load_heavy_model():
    time.sleep(3)                                   # 무거운 모델을 불러오는 척 3초 기다리기
    return {"name": "가짜 모델", "loaded_at": time.strftime("%H:%M:%S")}

model = load_heavy_model()                          # 처음만 3초, 그다음부터는 냉장고에서 바로 꺼내요
st.write("모델 로딩 시각(캐시라서 바뀌지 않음):", model["loaded_at"])

# ── ④ 편지 봉투(form) ────────────────────────────────────────
# 봉투 안에서 글자를 칠 때는 아무 일도 안 일어나고,
# '제출' 버튼(우체통)을 눌러야 그때 한 번 다시 실행돼요.
with st.form("demo_form"):                          # 여기서부터 봉투 안
    name = st.text_input("이름")                    # 이름 쓰는 칸
    submitted = st.form_submit_button("제출")       # 제출 버튼 (누르면 True)
if submitted:                                       # 제출했을 때만
    st.success(f"안녕하세요, {name}님!")            # 초록색 인사 상자 보여 주기
