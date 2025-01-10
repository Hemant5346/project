import os
from pathlib import Path
import re
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import uuid
import logging
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('upload.log')
    ]
)
logger = logging.getLogger(__name__)

class MarkdownProcessor:
    def __init__(self, model_name: str = 'intfloat/multilingual-e5-large',
                 qdrant_url: str = None,
                 api_key: str = None):
        """Initialize the Markdown processor with model and database connection"""
        # Use environment variables as fallback
        self.qdrant_url = qdrant_url or os.getenv('QDRANT_URL')
        self.api_key = api_key or os.getenv('QDRANT_API_KEY')
        
        if not self.qdrant_url or not self.api_key:
            raise ValueError("Qdrant Cloud URL and API key are required. Set them in .env file or pass as parameters.")
            
        self.model = SentenceTransformer(model_name)
        self.qdrant_client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.api_key
        )
        self.collection_name = "markdown_embeddings1"
        self.vector_size = 1024  # E5 model dimension
        
    def initialize_collection(self) -> bool:
        """Initialize or recreate the vector database collection"""
        try:
            # Delete collection if exists
            try:
                self.qdrant_client.delete_collection(self.collection_name)
                logger.info("Deleted existing collection")
            except Exception:
                pass
            
            # Create new collection
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Initialized collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error initializing collection: {str(e)}")
            raise

    def chunk_markdown(self, content: str, min_chunk_size: int = 200, 
                      max_chunk_size: int = 1000) -> List[str]:
        """Split markdown content into meaningful chunks while preserving context"""
        # Split content into sections based on headers
        sections = re.split(r'(?=#+\s)', content)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
                
            # Split section into paragraphs
            paragraphs = re.split(r'\n\s*\n', section)
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                    
                para_size = len(para.split())
                
                if current_size + para_size <= max_chunk_size:
                    current_chunk.append(para)
                    current_size += para_size
                else:
                    # If current chunk is too small, continue accumulating
                    if current_size < min_chunk_size and len(paragraphs) > 1:
                        continue
                    
                    # Save current chunk
                    if current_chunk:
                        chunks.append('\n\n'.join(current_chunk))
                    
                    # Start new chunk
                    current_chunk = [para]
                    current_size = para_size
        
        # Add final chunk if it exists and meets minimum size
        if current_chunk and current_size >= min_chunk_size:
            chunks.append('\n\n'.join(current_chunk))
            
        return chunks

    def process_markdown_file(self, file_path: str) -> bool:
        """Process a single markdown file and upload to vector database"""
        try:
            # Read markdown file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Get file name for metadata
            file_name = Path(file_path).name
            logger.info(f"Processing file: {file_name}")
            
            # Split into chunks
            chunks = self.chunk_markdown(content)
            logger.info(f"Created {len(chunks)} chunks")
            
            points = []
            for idx, chunk in enumerate(chunks):
                try:
                    # Create instruction-style text for better embedding
                    instruction_text = f"represent this presentation content for retrieval: {chunk}"
                    embedding = self.model.encode(instruction_text).tolist()
                    
                    # Create payload with metadata
                    payload = {
                        "text": chunk,
                        "chunk_id": idx,
                        "file_name": file_name,
                        "total_chunks": len(chunks)
                    }
                    
                    point = PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embedding,
                        payload=payload
                    )
                    points.append(point)
                    
                except Exception as e:
                    logger.error(f"Error processing chunk {idx}: {str(e)}")
                    continue
            
            # Upload points in batches
            if points:
                batch_size = 50
                for i in range(0, len(points), batch_size):
                    batch = points[i:i + batch_size]
                    self.qdrant_client.upsert(
                        collection_name=self.collection_name,
                        points=batch,
                        wait=True
                    )
                    logger.info(f"Uploaded batch {i//batch_size + 1} for {file_name}")
                
            logger.info(f"Successfully processed {file_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return False

def process_directory(directory_path: str, qdrant_url: str = None, api_key: str = None):
    """Process all markdown files in a directory"""
    try:
        # Initialize processor with cloud credentials
        processor = MarkdownProcessor(
            qdrant_url=qdrant_url,
            api_key=api_key
        )
        
        # Initialize collection
        processor.initialize_collection()
        
        # Get all markdown files
        markdown_files = list(Path(directory_path).rglob('*.md'))
        total_files = len(markdown_files)
        
        if total_files == 0:
            logger.warning(f"No markdown files found in {directory_path}")
            return
        
        logger.info(f"Found {total_files} markdown files")
        
        # Process each file with progress bar
        successful = 0
        failed = 0
        
        for file_path in tqdm(markdown_files, desc="Processing files"):
            try:
                if processor.process_markdown_file(str(file_path)):
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                failed += 1
                continue
        
        # Final summary
        logger.info("Processing complete!")
        logger.info(f"Successfully processed: {successful} files")
        logger.info(f"Failed to process: {failed} files")
        
    except Exception as e:
        logger.error(f"Directory processing error: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process markdown files for vector search")
    parser.add_argument("directory", type=str, help="Directory containing markdown files")
    parser.add_argument("--qdrant-url", type=str, 
                      help="Qdrant Cloud URL (or use QDRANT_URL in .env)",
                      default=None)
    parser.add_argument("--api-key", type=str,
                      help="Qdrant API Key (or use QDRANT_API_KEY in .env)",
                      default=None)
    
    args = parser.parse_args()
    
    print(f"\nProcessing markdown files from: {args.directory}")
    process_directory(args.directory, args.qdrant_url, args.api_key)