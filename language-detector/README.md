# Language Detector – Nhận diện ngôn ngữ

> **Lưu ý:** Đây chỉ là dự án thử nghiệm để tìm hiểu và làm quen với NLP, không phải sản phẩm hoàn chỉnh.

Dự án NLP nhỏ: nhập một câu bất kỳ, mô hình đoán câu đó thuộc ngôn ngữ nào (Việt, Anh, Pháp, Đức, Tây Ban Nha, Nhật).

## Cách hoạt động

1. **Dữ liệu**: các câu có gắn nhãn ngôn ngữ từ [Tatoeba](https://tatoeba.org/en/downloads).
2. **Vector hóa**: TF-IDF theo *character n-gram* (cụm 1–3 ký tự). Cách này hiệu quả cho nhận diện ngôn ngữ vì mỗi ngôn ngữ có các cụm ký tự đặc trưng (ví dụ dấu thanh tiếng Việt).
3. **Mô hình**: so sánh Naive Bayes và Logistic Regression (scikit-learn), chọn mô hình có accuracy cao hơn trên tập kiểm tra.
4. **Giao diện**: app web đơn giản bằng Streamlit.

## Cấu trúc thư mục

```
language-detector/
├── data/                 # file Tatoeba tải về (.tsv.bz2) và sentences.csv
├── prepare_data.py       # bước 2: gộp dữ liệu thành data/sentences.csv
├── train.py              # bước 3: huấn luyện, đánh giá, lưu model.joblib
├── app.py                # bước 4: app Streamlit
├── model.joblib          # mô hình đã huấn luyện (do train.py tạo ra)
└── README.md
```

## Yêu cầu

- Python 3.9 trở lên
- Thư viện: `pandas`, `scikit-learn`, `joblib`, `streamlit`

## Cài đặt

```bash
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install pandas scikit-learn joblib streamlit
```

## Cách chạy

**1. Tải dữ liệu.** Vào <https://tatoeba.org/en/downloads>, phần *Sentences*, chọn "Only sentences in" và tải từng ngôn ngữ muốn dùng (ví dụ Vietnamese, English, French, German, Spanish, Japanese). Đặt các file tải về vào thư mục `data/` (không cần giải nén).

**2. Gộp dữ liệu.**

```bash
python prepare_data.py
```

Mỗi ngôn ngữ lấy tối đa 5.000 câu (đổi `MAX_PER_LANG` trong file nếu muốn), kết quả ghi ra `data/sentences.csv`.

**3. Huấn luyện.**

```bash
python train.py
```

In ra accuracy, báo cáo theo từng ngôn ngữ, confusion matrix, rồi lưu mô hình vào `model.joblib`.

**4. Chạy app.**

```bash
streamlit run app.py
```

Mở `http://localhost:8501` trong trình duyệt.

## Kết quả

Dữ liệu gồm 6 ngôn ngữ (`deu`, `eng`, `fra`, `jpn`, `spa`, `vie`), mỗi ngôn ngữ 5.000 câu. Tập kiểm tra có 6.000 câu (1.000 câu mỗi ngôn ngữ).

| Mô hình | Accuracy |
|---|---|
| Naive Bayes (được chọn) | 0,9973 |
| Logistic Regression | 0,9967 |

Với Naive Bayes, mô hình đoán sai 16/6.000 câu:

- Tiếng Nhật và tiếng Tây Ban Nha đạt 1.000/1.000 câu đúng.
- Tiếng Việt chỉ sai 2 câu (998/1.000), tiếng Đức sai 1 câu (999/1.000).
- Nhầm nhiều nhất ở các ngôn ngữ dùng chữ Latin: tiếng Pháp bị đoán là tiếng Anh 4 câu, tiếng Anh bị đoán là tiếng Tây Ban Nha 3 câu, và một số cặp khác 2 câu.

Hai mô hình chênh nhau rất ít, nên kết luận "Naive Bayes tốt hơn" chưa chắc chắn. Độ chính xác cao này cũng phần nào do tập Tatoeba gồm các câu ngắn, sạch, nên chưa phản ánh văn bản ngoài đời thật.

## Hạn chế

- Câu rất ngắn (1–2 từ) có độ tin cậy thấp.
- Tiếng Việt không dấu ("toi dang tim viec") có thể bị nhận sai vì dữ liệu huấn luyện là tiếng Việt có dấu.
- Câu trộn nhiều ngôn ngữ chỉ cho ra một nhãn duy nhất.
- Mô hình chỉ biết các ngôn ngữ có trong dữ liệu huấn luyện; với ngôn ngữ lạ, nó vẫn đoán một trong các ngôn ngữ đã học.

## Hướng phát triển

- Thêm dữ liệu tiếng Việt không dấu (bỏ dấu tự động từ dữ liệu có sẵn).
- Thêm nhiều ngôn ngữ hơn.
- So sánh với `langdetect` hoặc `fastText`.
- Thử các giá trị `ngram_range` khác và đo ảnh hưởng đến accuracy.

## Nguồn dữ liệu và giấy phép

Dữ liệu câu lấy từ [Tatoeba](https://tatoeba.org), phát hành theo giấy phép [CC BY 2.0 FR](https://creativecommons.org/licenses/by/2.0/fr/). Nếu chia sẻ dự án công khai, hãy ghi nguồn Tatoeba.