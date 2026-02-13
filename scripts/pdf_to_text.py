from pypdf import PdfReader

reader = PdfReader("data/520001m_vol1.pdf")
texts = []
for i, page in enumerate(reader.pages):
    t = page.extract_text() or ""
    texts.append(t)

full_text = "\n\n".join(texts)
open("data/520001m_vol1.txt", "w", encoding="utf-8").write(full_text)