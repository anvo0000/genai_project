from langchain_community.document_loaders import PyPDFLoader

pdf_loader = PyPDFLoader("../files/MLA-C01.pdf")
docs = pdf_loader.load()
for doc in docs:
    print(doc.page_content)