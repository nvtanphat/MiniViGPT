import pytest
pytest.importorskip("tokenizers")

from minivigpt.tokenizer import load_tokenizer, train_bpe_tokenizer
from minivigpt.tokenizer_eval import evaluate_texts


def test_byte_bpe_roundtrip_vietnamese_and_zero_unk(tmp_path):
    corpus = [
        "Việt Nam là một quốc gia ở Đông Nam Á.",
        "Trí tuệ nhân tạo đang thay đổi cách chúng ta làm việc.",
        "Mô hình ngôn ngữ học cách dự đoán token tiếp theo.",
    ] * 50
    path = tmp_path / "tokenizer.json"
    tok = train_bpe_tokenizer(corpus, path, vocab_size=512, min_frequency=1)
    tok = load_tokenizer(path)
    text = "Tiếng Việt có dấu: học máy, dữ liệu và trí tuệ nhân tạo."
    encoded = tok.encode(text)
    decoded = tok.decode(encoded.ids, skip_special_tokens=True)
    assert decoded == text
    stats = evaluate_texts(tok, [text])
    assert stats["unk_rate"] == 0.0
