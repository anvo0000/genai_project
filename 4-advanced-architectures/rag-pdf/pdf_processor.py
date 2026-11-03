import uuid
import PyPDF2

## CONSTANTS
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200 # keeps context between chunks

class PDFProcessor:
    """Handle PDF files and chunking text"""
    def __init__(self, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_over_lap = chunk_overlap

    @staticmethod
    def read_pdf(pdf_file):
        reader = PyPDF2.PdfReader(pdf_file)
        _text = ""
        for page in reader.pages:
            _text += page.extract_text() + "\n"
        return _text

    def create_chunk(self, pdf_text, pdf_file):
        chunks = []
        start = 0

        source_name = pdf_file
        if hasattr(pdf_file, 'name'):  # If it's an UploadedFile object
            source_name = pdf_file.name

        while start < len(pdf_text):
            end = start + self.chunk_size

            # if this is in the middle of the text, then combine with chunk_overlap
            if start > 0:
                start = start - self.chunk_over_lap

            _chunk = pdf_text[start: end]

            # Try to break at the sentence ending (.)
            if end < len(pdf_text):
                last_period = _chunk.rfind(".")
                if last_period != -1:
                    _chunk = _chunk[: last_period + 1]
                    end = start + last_period + 1
            chunks.append(
                {
                    "id": str(uuid.uuid4()),
                    "text": _chunk,
                    "metadata": {"source": source_name}
                }
            )
            start = end
        return chunks
