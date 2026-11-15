import sys
from pypdf import PdfReader

# Usage :
#   python src\peek_pdf.py fichier.pdf 10          -> affiche la page 10
#   python src\peek_pdf.py fichier.pdf franchise   -> liste les pages contenant "franchise"
fichier, arg = sys.argv[1], sys.argv[2]
reader = PdfReader(fichier)

if arg.isdigit():
    print(reader.pages[int(arg) - 1].extract_text()[:1500])
else:
    for i, page in enumerate(reader.pages, start=1):
        texte = (page.extract_text() or "").lower()
        if arg.lower() in texte:
            print(f"page {i} : {texte.count(arg.lower())} occurrence(s)")