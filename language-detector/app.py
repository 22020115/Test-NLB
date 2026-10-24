"""Buoc 4: app Streamlit nhan dien ngon ngu. Chay: streamlit run app.py"""
from pathlib import Path

import joblib
import streamlit as st

MODEL_PATH = Path(__file__).parent / "model.joblib"

# Ten day du cho ma ngon ngu ISO 639-3 cua Tatoeba (ma nao khong co se hien nguyen ma)
LANG_NAMES = {
    "vie": "Tiếng Việt",
    "eng": "Tiếng Anh",
    "fra": "Tiếng Pháp",
    "deu": "Tiếng Đức",
    "spa": "Tiếng Tây Ban Nha",
    "jpn": "Tiếng Nhật",
    "cmn": "Tiếng Trung (Quan thoại)",
    "rus": "Tiếng Nga",
    "ita": "Tiếng Ý",
    "por": "Tiếng Bồ Đào Nha",
    "kor": "Tiếng Hàn",
}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.set_page_config(page_title="Nhận diện ngôn ngữ", page_icon="🌐")
st.title("🌐 Nhận diện ngôn ngữ")
st.write("Nhập một câu bất kỳ, mô hình sẽ đoán đó là ngôn ngữ nào.")

if not MODEL_PATH.exists():
    st.error("Chưa có model.joblib. Hãy chạy `python train.py` trước.")
    st.stop()

model = load_model()

text = st.text_area("Câu cần nhận diện", height=120, placeholder="Ví dụ: Tôi đang tìm việc làm")

if st.button("Nhận diện", type="primary"):
    if not text.strip():
        st.warning("Bạn chưa nhập câu nào.")
    else:
        probs = model.predict_proba([text.lower()])[0]
        ranked = sorted(zip(model.classes_, probs), key=lambda x: x[1], reverse=True)

        best_code, best_p = ranked[0]
        st.success(f"Kết quả: **{LANG_NAMES.get(best_code, best_code)}** ({best_p:.1%})")
        if best_p < 0.6:
            st.info("Độ tin cậy thấp: câu có thể quá ngắn, lẫn nhiều ngôn ngữ, hoặc ngoài các ngôn ngữ đã học.")

        st.subheader("Top 3 khả năng")
        for code, p in ranked[:3]:
            st.write(f"{LANG_NAMES.get(code, code)}: {p:.1%}")
            st.progress(float(p))