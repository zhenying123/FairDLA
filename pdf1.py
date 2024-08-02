from PIL import Image

def png_to_pdf(png_path, pdf_path):
    image = Image.open(png_path)
    image = image.convert('RGB')  # Ensure the image is in RGB mode
    image.save(pdf_path, "PDF", resolution=100.0)

# # Example usage:
# png_path = '/home/yzhen/code/fair/FairSAD_copy/combined_plot_pokecn.png'
# pdf_path = '/home/yzhen/code/fair/FairSAD_copy/combined_plot_pokecn.pdf'
# png_to_pdf(png_path, pdf_path)
def png_to_pdf(png_path, pdf_path):
    image = Image.open(png_path)
    image = image.convert('RGB')  # Ensure the image is in RGB mode
    image.save(pdf_path, "PDF", resolution=100.0)

# Example usage:
png_path = '/home/yzhen/code/fair/FairSAD_copy/combined_plot_keshihua.png'
pdf_path = '/home/yzhen/code/fair/FairSAD_copy/combined_plot_keshihua.pdf'
png_to_pdf(png_path, pdf_path)