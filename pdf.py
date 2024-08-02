import glob
import fitz
import os

def pic2pdf(pdf_name, pic_folder):
    doc = fitz.open()
    for img in sorted(glob.glob(os.path.join(pic_folder, "*.png"))):  # Read images, ensure sorted by filename
        print(img)
        imgdoc = fitz.open(img)  # Open image
        pdfbytes = imgdoc.convert_to_pdf()  # Create single-page PDF from image
        imgpdf = fitz.open("pdf", pdfbytes)
        doc.insert_pdf(imgpdf)  # Insert current page into document
    
    # Correct the PDF filename
    if not pdf_name.endswith(".pdf"):
        pdf_name += ".pdf"
    
    # Save in the image folder
    save_pdf_path = os.path.join(pic_folder, pdf_name)
    if os.path.exists(save_pdf_path):
        os.remove(save_pdf_path)
    
    doc.save(save_pdf_path)  # Save PDF file
    doc.close()




if __name__ == '__main__':
    pic2pdf("plot_prob.pdf", "/home/yzhen/code/fair/FairSAD_copy/prob")
