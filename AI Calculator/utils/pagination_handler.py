from pdf2image import convert_from_path

def handle_pagination(file):
    # Handle multi-page PDF to images
    if file.name.endswith('.pdf'):
        pages = convert_from_path(file)
        return pages
    else:
        return [file]
