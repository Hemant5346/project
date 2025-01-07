# search_app.py

import streamlit as st
import os
from typing import List
from dotenv import load_dotenv
import re
import requests
from typing import Optional

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
        """Initialize searcher with API endpoint settings"""
        self.api_endpoint = os.getenv("SEARCH_API_ENDPOINT")
        self.api_key = os.getenv("API_KEY")
        
        if not self.api_endpoint:
            raise ValueError("Search API endpoint not found in environment variables")

    def search(self, query: str, top_k: int = 3) -> List[SearchResult]:
        """Perform semantic search via API"""
        try:
            # Prepare API request
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
                
            params = {
                "query": query,
                "top_k": top_k
            }
            
            # Make API request
            response = requests.post(
                self.api_endpoint,
                json=params,
                headers=headers
            )
            response.raise_for_status()
            
            # Process results
            results = response.json()
            return [
                SearchResult(
                    text=item["text"],
                    document=item["document_name"],
                    score=item["score"],
                    page=item.get("page_number", 1)
                )
                for item in results
            ]
            
        except Exception as e:
            st.error(f"Search error: {str(e)}")
            return []

@st.cache_resource
def get_searcher() -> Optional[DocumentSearcher]:
    """Initialize and cache the searcher"""
    try:
        return DocumentSearcher()
    except ValueError as e:
        st.error(str(e))
        return None

def main():
    st.set_page_config(page_title="Financial Document Search", page_icon="📄")
    
    st.title("Financial Document Search System")
    st.write("Search through your uploaded financial documents using semantic search.")

    # Initialize searcher
    searcher = get_searcher()
    if not searcher:
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
        "The search is powered by a separate API service that handles "
        "the document embeddings and vector search."
    )

if __name__ == "__main__":
    main()