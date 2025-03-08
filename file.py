import os
import glob
from PyPDF2 import PdfReader
from docx import Document

# Set the directory where Google Drive files are stored
drive_folder = "foodfixrgd"

# Create an output folder for .txt files
output_folder = os.path.join(drive_folder, "txt_files")
os.makedirs(output_folder, exist_ok=True)

# Convert .docx to .txt
def convert_docx_to_txt(docx_path, txt_path):
    doc = Document(docx_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)

# Convert .pdf to .txt
def convert_pdf_to_txt(pdf_path, txt_path):
    reader = PdfReader(pdf_path)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)

# Convert all files
for file_path in glob.glob(drive_folder + "/**/*", recursive=True):
    filename, ext = os.path.splitext(file_path)
    if ext.lower() == ".docx":
        convert_docx_to_txt(file_path, os.path.join(output_folder, os.path.basename(filename) + ".txt"))
    elif ext.lower() == ".pdf":
        convert_pdf_to_txt(file_path, os.path.join(output_folder, os.path.basename(filename) + ".txt"))
    elif ext.lower() == ".txt":
        # Copy existing .txt files
        os.rename(file_path, os.path.join(output_folder, os.path.basename(file_path)))

print("✅ All files converted to .txt and saved in", output_folder)
