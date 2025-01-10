import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from typing import List, Dict
import logging
from os import environ
from dotenv import load_dotenv
import os


# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('search.log')
    ]
)
logger = logging.getLogger(__name__)

class SearchResult:
    """Class to hold search result information"""
    def __init__(self, text: str, file_name: str, chunk_id: int, score: float):
        self.text = text
        self.file_name = file_name
        self.chunk_id = chunk_id
        self.score = score

def display_results(results: List[SearchResult]):
    """Display search results in a formatted way"""
    for idx, result in enumerate(results, 1):
        st.markdown(f"### Result {idx}")
        with st.expander(f"Score: {result.score:.3f} - {result.file_name}"):
            st.markdown(f"**Source File:** {result.file_name}")
            st.markdown(f"**Chunk ID:** {result.chunk_id}")
            st.markdown(f"**Relevance Score:** {result.score:.3f}")
            st.markdown("**Content:**")
            st.markdown(result.text)
        st.markdown("---")

class MarkdownSearcher:
    def __init__(self, model_name: str = 'intfloat/multilingual-e5-large',
                 qdrant_url: str = None,
                 api_key: str = None):
        """Initialize the Markdown searcher with model and database connection"""
        # Use provided credentials or fall back to environment variables
        self.qdrant_url = qdrant_url or os.getenv('QDRANT_URL')
        self.api_key = api_key or os.getenv('QDRANT_API_KEY')
        
        if not self.qdrant_url or not self.api_key:
            raise ValueError("Qdrant Cloud URL and API key are required. Set them in .env file or pass as parameters.")
            
        try:
            self.model = SentenceTransformer(model_name)
            self.qdrant_client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.api_key
            )
            logger.info(f"Successfully connected to Qdrant Cloud at {self.qdrant_url}")
        except Exception as e:
            logger.error(f"Failed to initialize: {str(e)}")
            raise
        self.collection_name = "markdown_embeddings1"

    def verify_collection(self) -> bool:
        """Verify that the collection exists and contains data"""
        try:
            # Get collections list
            collections_response = self.qdrant_client.get_collections()
            
            # Log raw response for debugging
            logger.info(f"Raw collections response: {collections_response}")
            
            # Get list of collection names
            collections = collections_response.collections
            collection_names = [collection.name for collection in collections]
            
            logger.info(f"Found collections: {collection_names}")
            
            if self.collection_name not in collection_names:
                logger.warning(f"Collection {self.collection_name} not found in {collection_names}")
                return False
            
            # Get specific collection info
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            
            if collection_info.vectors_count == 0:
                logger.warning(f"Collection {self.collection_name} exists but is empty")
                return False
            
            logger.info(f"Verified collection {self.collection_name} with {collection_info.vectors_count} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Error verifying collection: {str(e)}")
            logger.exception("Full traceback:")
            return False

    def search(self, query: str, top_k: int = 3) -> List[SearchResult]:
        """Search for relevant content using the query"""
        if not query.strip():
            return []
            
        try:
            # Verify collection exists and has data
            if not self.verify_collection():
                raise Exception("No data available for search. Please ensure documents are uploaded.")
            
            # Create instruction-style query
            instruction_query = f"retrieve presentation content about: {query}"
            query_vector = self.model.encode(instruction_query).tolist()
            
            # Search in Qdrant
            results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k
            )
            
            # Process results
            search_results = []
            for res in results:
                result = SearchResult(
                    text=res.payload["text"],
                    file_name=res.payload["file_name"],
                    chunk_id=res.payload["chunk_id"],
                    score=res.score
                )
                search_results.append(result)
            
            return search_results
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            raise

def main():
    st.title("Markdown Content Search System")
    
    # Get Qdrant Cloud credentials from environment variables
    qdrant_url = os.getenv('QDRANT_URL')
    api_key = os.getenv('QDRANT_API_KEY')
    
    if not qdrant_url or not api_key:
        st.error("Qdrant Cloud credentials not found. Please ensure QDRANT_URL and QDRANT_API_KEY are set in your .env file.")
        return
    
    # Initialize searcher in session state with error handling
    if 'searcher' not in st.session_state:
        try:
            logger.info("Initializing new MarkdownSearcher...")
            st.session_state.searcher = MarkdownSearcher(
                qdrant_url=qdrant_url,
                api_key=api_key
            )
            logger.info("MarkdownSearcher initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MarkdownSearcher: {str(e)}")
            st.error("Failed to initialize search system. Please check your connection settings.")
            return

    # Add debug information
    with st.sidebar:
        st.markdown("### System Status")
        try:
            collection_status = st.session_state.searcher.verify_collection()
            st.write("Database Connection:", "✅ Connected" if collection_status else "❌ Not Connected")
            if not collection_status:
                st.warning("Collection not found or empty")
        except Exception as e:
            st.error(f"Error checking status: {str(e)}")

    # Search interface elements
    st.markdown("### Search Your Markdown Content")
    query = st.text_input("Enter your search query:")
    top_k = st.slider("Number of results to show", min_value=1, max_value=10, value=3)

    if query:
        try:
            # Verify collection before searching
            collection_status = st.session_state.searcher.verify_collection()
            
            if not collection_status:
                st.warning("Database is not properly initialized or empty. Please ensure your documents are uploaded.")
                return

            # Perform search
            results = st.session_state.searcher.search(query, top_k=top_k)
            
            if results:
                st.markdown("### Search Results")
                display_results(results)
            else:
                st.info("No matching content found for your query.")
                
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            st.error(f"An error occurred during search: {str(e)}")

if __name__ == "__main__":
    main()