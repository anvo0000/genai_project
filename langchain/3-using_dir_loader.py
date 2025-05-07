from langchain_community.document_loaders import DirectoryLoader

dir_loader = DirectoryLoader(path="../files/", glob="**/*.txt")
dir_documents = dir_loader.load()

# print(dir_documents)
for doc in dir_documents:
    metadata = doc.metadata
    content = doc.page_content
    print(metadata)
    print(content[:100])