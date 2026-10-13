"""
Script chuyển đổi Tai_Lieu_Thuyet_Trinh_Va_Cau_Hoi_Phan_Bien.md sang HTML và PDF
với công thức toán MathML sắc nét, mục lục liên kết Clickable GoTo PDF Links và trang bìa trang trọng.
"""

import os
import sys
import re
import subprocess
import bs4
import markdown
import pymupdf
import latex2mathml.converter

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MD_PATH = os.path.join(CURRENT_DIR, "Tai_Lieu_Thuyet_Trinh_Va_Cau_Hoi_Phan_Bien.md")
HTML_PATH = os.path.join(CURRENT_DIR, "Tai_Lieu_Thuyet_Trinh_Va_Cau_Hoi_Phan_Bien.html")
PDF_PATH = os.path.join(CURRENT_DIR, "Tai_Lieu_Thuyet_Trinh_Va_Cau_Hoi_Phan_Bien.pdf")

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
EDGE_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
BROWSER_EXE = CHROME_PATH if os.path.exists(CHROME_PATH) else EDGE_PATH

def convert_latex_to_mathml_in_text(md_text):
    def replace_block(match):
        raw = match.group(1).strip()
        try:
            m = latex2mathml.converter.convert(raw, display="block")
            return f"\n\n<div class=\"math-block\">{m}</div>\n\n"
        except Exception:
            return f"\n\n<div class=\"math-block\"><pre class=\"math-raw\">{raw}</pre></div>\n\n"

    def replace_inline(match):
        raw = match.group(1).strip()
        try:
            m = latex2mathml.converter.convert(raw, display="inline")
            return f"<span class=\"math-inline\">{m}</span>"
        except Exception:
            return f"<code class=\"math-raw-inline\">{raw}</code>"

    text = re.sub(r'\$\$(.*?)\$\$', replace_block, md_text, flags=re.DOTALL)
    text = re.sub(r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)', replace_inline, text)
    return text

def make_html():
    with open(MD_PATH, "r", encoding="utf-8") as f:
        md_text = f.read()

    md_with_mathml = convert_latex_to_mathml_in_text(md_text)
    md_parser = markdown.Markdown(extensions=['tables', 'fenced_code'])
    html_body = md_parser.convert(md_with_mathml)

    soup = bs4.BeautifulSoup(html_body, 'html.parser')

    headings = []
    for el in soup.find_all(['h1', 'h2', 'h3', 'h4']):
        if "MỤC LỤC TỔNG QUAN" in el.text.upper():
            continue
        headings.append(el)

    heading_lookup = {}
    for idx, h in enumerate(headings):
        sec_id = f"pres-sec-{idx+1}"
        h['id'] = sec_id
        clean_text = re.sub(r'[^a-zA-Z0-9\u00C0-\u1EF9]', '', h.text.lower())
        heading_lookup[clean_text] = sec_id

    for a in soup.find_all('a'):
        raw_text = a.text
        clean_link = re.sub(r'[^a-zA-Z0-9\u00C0-\u1EF9]', '', raw_text.lower())
        matched_id = None
        for h_text, h_id in heading_lookup.items():
            if clean_link == h_text or clean_link in h_text or h_text in clean_link:
                matched_id = h_id
                break
        if matched_id:
            a['href'] = f"#{matched_id}"

    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <title>Cẩm Nang Thuyết Trình & Bộ 30 Câu Hỏi Phản Biện AI SIC</title>
    <style>
        @page {{
            size: A4;
            margin: 20mm 18mm 20mm 18mm;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            font-size: 10pt;
            line-height: 1.6;
            color: #1E293B;
            margin: 0;
            padding: 0;
        }}
        .cover-page {{
            page-break-after: always;
            height: 90vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
        }}
        .cover-badge {{
            display: inline-block;
            background: #EFF6FF;
            color: #2563EB;
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 11pt;
            font-weight: 700;
            letter-spacing: 1.5px;
            margin-bottom: 24px;
            border: 1px solid #BFDBFE;
        }}
        .cover-title {{
            font-size: 32pt;
            font-weight: 900;
            color: #0F172A;
            line-height: 1.25;
            margin-bottom: 20px;
            max-width: 800px;
        }}
        .cover-subtitle {{
            font-size: 16pt;
            font-weight: 500;
            color: #64748B;
            margin-bottom: 35px;
            max-width: 700px;
        }}
        .cover-author {{
            font-size: 18pt;
            font-weight: 700;
            color: #2563EB;
        }}
        .cover-org {{
            font-size: 13pt;
            color: #475569;
            margin-top: 6px;
        }}
        h1 {{
            font-size: 17pt;
            color: #0F172A;
            border-bottom: 2px solid #2563EB;
            padding-bottom: 6px;
            margin-top: 32px;
            margin-bottom: 14px;
            page-break-after: avoid;
        }}
        h2 {{
            font-size: 13.5pt;
            color: #1E3A8A;
            margin-top: 24px;
            margin-bottom: 10px;
            page-break-after: avoid;
        }}
        h3 {{
            font-size: 11.5pt;
            color: #047857;
            margin-top: 18px;
            margin-bottom: 8px;
            page-break-after: avoid;
        }}
        h4 {{
            font-size: 10.5pt;
            color: #B45309;
            margin-top: 14px;
            margin-bottom: 6px;
            page-break-after: avoid;
        }}
        p, li {{
            text-align: justify;
        }}
        a {{
            color: #2563EB;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        blockquote {{
            border-left: 4px solid #2563EB;
            margin: 10px 0;
            padding: 8px 16px;
            background-color: #F8FAFC;
            color: #0F172A;
            border-radius: 0 8px 8px 0;
            font-style: normal;
            page-break-inside: avoid;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 14px 0;
            font-size: 9pt;
            page-break-inside: avoid;
        }}
        th, td {{
            border: 1px solid #CBD5E1;
            padding: 7px 10px;
            text-align: left;
        }}
        th {{
            background-color: #F1F5F9;
            color: #0F172A;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #F8FAFC;
        }}
        code {{
            background-color: #F1F5F9;
            padding: 2px 5px;
            border-radius: 4px;
            font-family: Consolas, monospace;
            font-size: 9pt;
            color: #BE185D;
        }}
        math {{
            font-family: "Cambria Math", "Latin Modern Math", STIXGeneral, serif;
            font-size: 1.15em;
        }}
        .math-block {{
            display: block;
            text-align: center;
            margin: 14px auto;
            padding: 10px 14px;
            background: #F8FAFC;
            border-radius: 8px;
            border: 1px solid #E2E8F0;
            page-break-inside: avoid;
            overflow-x: auto;
        }}
        .math-inline {{
            display: inline-block;
            padding: 0 2px;
        }}
        hr {{
            border: none;
            border-top: 1px solid #E2E8F0;
            margin: 20px 0;
        }}
    </style>
</head>
<body>

<div class="cover-page">
    <div class="cover-badge">SAMSUNG INNOVATION CAMPUS (AI SIC)</div>
    <div class="cover-title">KỊCH BẢN THUYẾT TRÌNH 10 PHÚT & BỘ DỰ ĐOÁN 30 CÂU HỎI PHẢN BIỆN</div>
    <div class="cover-subtitle">Đồ án Capstone: Hệ thống Gợi ý Phối đồ Thời trang Thông minh ứng dụng Deep Learning</div>
    <div class="cover-author">Đoàn Mạnh Dũng</div>
    <div class="cover-org">Chuyên ban Computer Vision & Recommender Systems</div>
</div>

<div class="content-body">
{str(soup)}
</div>

</body>
</html>"""

    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"✅ Đã tạo file HTML: {HTML_PATH}")

def convert_to_pdf():
    cmd = [
        BROWSER_EXE,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={PDF_PATH}",
        HTML_PATH
    ]
    print(f"🖨️ Đang xuất PDF qua Chrome Headless: {PDF_PATH}...")
    subprocess.run(cmd, check=True)

    doc = pymupdf.open(PDF_PATH)
    total_links = sum(len(p.get_links()) for p in doc)
    print(f"📄 File: Tai_Lieu_Thuyet_Trinh_Va_Cau_Hoi_Phan_Bien.pdf | Tổng số trang: {len(doc)} | Tổng liên kết Clickable: {total_links}")

if __name__ == "__main__":
    make_html()
    convert_to_pdf()
