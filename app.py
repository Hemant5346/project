#search_app.py

import streamlit as st
import os
from typing import List
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

class SearchResult:
    def __init__(self, text: str, document: str, score: float, page: int):
        self.text = text
        self.document = document
        self.score = score
        self.page = page

class DocumentSearcher:
    def __init__(self):
        """Initialize searcher with Qdrant Cloud settings"""
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        if not self.qdrant_url or not self.qdrant_api_key:
            raise ValueError("Qdrant Cloud credentials not found in environment variables")
            
        self.model = SentenceTransformer('intfloat/multilingual-e5-large')
        self.qdrant_client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key
        )
        self.collection_name = "document_embeddings_v1"

    def search(self, query: str, top_k: int = 3) -> List[SearchResult]:
        """Perform semantic search"""
        try:
            # Create instruction-style query
            instruction_query = f"retrieve financial document information about: {query}"
            query_vector = self.model.encode(instruction_query).tolist()
            
            # Create search filter
            query_lower = query.lower()
            filter_conditions = []
            
            if re.search(r'\b20[12]\d\b', query_lower):
                filter_conditions.append(
                    FieldCondition(key="has_year", match=MatchValue(value=True))
                )
            if 'dividend' in query_lower:
                filter_conditions.append(
                    FieldCondition(key="has_dividend", match=MatchValue(value=True))
                )
            
            search_filter = Filter(should=filter_conditions) if filter_conditions else None
            
            # Perform search
            results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=search_filter
            )
            
            # Process results
            return [
                SearchResult(
                    text=res.payload["text"],
                    document=res.payload["document_name"],
                    score=res.score,
                    page=res.payload.get("page_number", 1)
                )
                for res in results
            ]
            
        except Exception as e:
            st.error(f"Search error: {str(e)}")
            return []

def main():
    st.set_page_config(page_title="Financial Document Search", page_icon="📄")
    
    st.title("Financial Document Search System")
    st.write("Search through your uploaded financial documents using semantic search.")

    # Initialize searcher
    @st.cache_resource
    def get_searcher():
        return DocumentSearcher()
    
    try:
        searcher = get_searcher()
    except ValueError as e:
        st.error(str(e))
        st.stop()

    # Search interface
    st.subheader("Search Documents")
    col1, col2 = st.columns([4, 1])
    
    with col1:
        query = st.text_input("Enter your search query:")
    with col2:
        top_k = st.number_input("Number of results:", min_value=1, max_value=10, value=3)

    if st.button("Search"):
        if query:
            with st.spinner("Searching..."):
                results = searcher.search(query, top_k=top_k)
            
            if results:
                st.write(f"Found {len(results)} results:")
                for result in results:
                    with st.expander(f"Score: {result.score:.3f} - {result.document}"):
                        st.markdown(f"**Page:** {result.page}")
                        st.markdown(f"**Text:**\n{result.text}")
            else:
                st.warning("No results found.")
        else:
            st.warning("Please enter a search query.")

    # Add information about data upload
    st.sidebar.title("About")
    st.sidebar.info(
        "This is a search interface for your financial documents. "
        "To upload new documents, please use the separate upload script "
        "with your Qdrant Cloud credentials."
    )

if __name__ == "__main__":
    main()