from pathlib import Path
from pypdf import PdfReader

DOSSIER_CG = Path("data/cg")

print(f"{'Fichier':45} {'Pages':>6} {'Caract./page':>13}  Statut")
print("-" * 80)

for pdf in sorted(DOSSIER_CG.glob("*.pdf")):
    reader = PdfReader(pdf)
    nb_pages = len(reader.pages)

    # On extrait le texte de chaque page et on compte les caractères
    total_caracteres = sum(len(page.extract_text() or "") for page in reader.pages)
    moyenne = total_caracteres / nb_pages if nb_pages else 0

    # Une page de CG contient normalement plus de 1000 caractères.
    # Presque rien = PDF scanné (une image), donc illisible sans OCR.
    statut = "OK" if moyenne > 500 else "⚠️ peu de texte est ce que c'est un PDF scanné ?"
    print(f"{pdf.name:45} {nb_pages:>6} {moyenne:>13.0f}  {statut}")