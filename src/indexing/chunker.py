from langchain_text_splitters import RecursiveCharacterTextSplitter

class Chunker:
    def __init__(self, chunk_size=800, chunk_overlap=100):
        """
        Initializes the Recursive Chunking logic.
        
        Args:
            chunk_size (int): Max characters per chunk. 
                              800 chars is roughly 150-200 words.
            chunk_overlap (int): How many characters to repeat between chunks 
                                 to preserve context.
        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            # Priority: Paragraphs -> Lines -> Sentences -> Words
            separators=["\n\n", "\n", ". ", " ", ""] 
        )

    def chunk_text(self, text):
        """
        Splits text into semantically meaningful chunks.
        Returns: list of strings.
        """
        if not text:
            return []
        
        docs = self.splitter.create_documents([text])
        return [d.page_content for d in docs]