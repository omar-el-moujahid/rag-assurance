import sys
from pypdf import PdfReader

# Usage : python src\peek_pdf.py data\cg\gmf_domopass_cg_2023-01.pdf 10
fichier, num_page = sys.argv[1], int(sys.argv[2])
reader = PdfReader(fichier)
print(reader.pages[num_page - 1].extract_text()[:1500])