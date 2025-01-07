
import os
import uuid
import re
import click
import nltk
import logging
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from tqdm import tqdm

# Load environment variables
load_dotenv()

class QdrantUploader:
    def __init__(self):
        """Initialize uploader with Qdrant Cloud settings"""
        self._setup_logging()
        self._setup_environment()
        self._setup_nltk()
        self._initialize_model()
        self._initialize_qdrant()

    def _setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('upload.log')
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _setup_environment(self):
        """Setup environment variables"""
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        if not self.qdrant_url or not self.qdrant_api_key:
            raise ValueError("Qdrant Cloud credentials not found in environment variables")

    def _setup_nltk(self):
        """Setup NLTK resources"""
        nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
        os.makedirs(nltk_data_path, exist_ok=True)
        nltk.data.path.append(nltk_data_path)
        
        resources = ['punkt', 'punkt_tab', 'averaged_perceptron_tagger', 'words']
        for resource in resources:
            try:
                nltk.download(resource, download_dir=nltk_data_path, quiet=True)
            except Exception as e:
                self.logger.warning(f"Failed to download {resource}: {str(e)}")
                
        # Verify punkt is properly loaded
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            # If still not found, try downloading directly to the default location
            nltk.download('punkt')

    def _initialize_model(self):
        """Initialize the sentence transformer model"""
        model_name = 'intfloat/multilingual-e5-large'
        self.model = SentenceTransformer(model_name)
        self.vector_size = 1024  # E5 model dimension

    def _initialize_qdrant(self):
        """Initialize Qdrant client with cloud credentials"""
        self.qdrant_client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            timeout=60
        )
        self.collection_name = "document_embeddings_v1"

    def create_collection(self, recreate: bool = False):
        """Create or recreate the collection"""
        try:
            # Delete if recreate is True and collection exists
            if recreate:
                try:
                    self.qdrant_client.delete_collection(self.collection_name)
                    self.logger.info(f"Deleted existing collection: {self.collection_name}")
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
            self.logger.info(f"Created collection: {self.collection_name}")
            return True
        except Exception as e:
            self.logger.error(f"Error creating collection: {str(e)}")
            return False

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        text = text.replace('"', '"').replace('"', '"').replace('—', '-')
        text = re.sub(r'[^\w\s.!?$%,()"-]', ' ', text)
        return ' '.join(text.split())

    def _extract_metadata(self, text: str) -> Dict:
        """Extract metadata from text"""
        text_lower = text.lower()
        return {
            'has_numbers': bool(re.search(r'\d+', text)),
            'has_percentage': bool(re.search(r'\d+(\.\d+)?%', text)),
            'has_currency': bool(re.search(r'(\$|€|£|aed)\s*\d+', text_lower)),
            'has_year': bool(re.search(r'\b20[12]\d\b', text)),
            'has_growth': bool(re.search(r'growth|increase|decrease|cagr', text_lower)),
            'has_financial': bool(re.search(r'revenue|profit|ebitda|margin|dividend', text_lower)),
            'has_revenue': 'revenue' in text_lower,
            'has_profit': 'profit' in text_lower or 'earnings' in text_lower,
            'has_dividend': 'dividend' in text_lower
        }

    def process_document(self, file_path: str) -> bool:
        """Process a single document and upload to Qdrant Cloud"""
        try:
            # First check if collection exists, create if it doesn't
            collections = self.qdrant_client.get_collections()
            if self.collection_name not in [c.name for c in collections.collections]:
                self.logger.info(f"Collection {self.collection_name} not found. Creating...")
                if not self.create_collection():
                    raise ValueError("Failed to create collection")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            except UnicodeDecodeError:
                with open(file_path, 'r', encoding='latin-1') as f:
                    text = f.read()

            # Clean and split text
            text = self._clean_text(text)
            sentences = nltk.sent_tokenize(text)
            
            # Process in chunks
            chunk_size = 3  # Number of sentences per chunk
            chunks = [sentences[i:i + chunk_size] for i in range(0, len(sentences), chunk_size)]
            
            points = []
            for idx, chunk in enumerate(chunks):
                chunk_text = ' '.join(chunk)
                instruction_text = f"represent this financial document text for retrieval: {chunk_text}"
                embedding = self.model.encode(instruction_text).tolist()
                
                payload = {
                    "text": chunk_text,
                    "document_name": os.path.basename(file_path),
                    "page_number": idx + 1,
                    **self._extract_metadata(chunk_text)
                }
                
                points.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload=payload
                ))

            # Upload points in batches with progress bar
            batch_size = 50
            total_batches = (len(points) + batch_size - 1) // batch_size
            
            with tqdm(total=total_batches, desc=f"Uploading {os.path.basename(file_path)}") as pbar:
                for i in range(0, len(points), batch_size):
                    batch = points[i:i + batch_size]
                    self.qdrant_client.upsert(
                        collection_name=self.collection_name,
                        points=batch,
                        wait=True
                    )
                    pbar.update(1)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing document {file_path}: {str(e)}")
            if "Collection doesn't exist" in str(e):
                self.logger.info("Trying to create collection and retry...")
                if self.create_collection():
                    # Retry the upload
                    return self.process_document(file_path)
            return False

    def process_directory(self, directory_path: str) -> Dict[str, int]:
        """Process all documents in a directory"""
        stats = {"success": 0, "failed": 0, "total": 0}
        
        try:
            dir_path = Path(directory_path)
            if not dir_path.exists():
                raise ValueError(f"Directory not found: {directory_path}")

            # Get all text and markdown files
            files = list(dir_path.glob('*.txt')) + list(dir_path.glob('*.md'))
            stats["total"] = len(files)

            if not files:
                self.logger.warning(f"No .txt or .md files found in {directory_path}")
                return stats

            # Process each file
            for file_path in files:
                if self.process_document(str(file_path)):
                    stats["success"] += 1
                else:
                    stats["failed"] += 1

        except Exception as e:
            self.logger.error(f"Directory processing error: {str(e)}")
            
        return stats

@click.group()
def cli():
    """CLI for uploading documents to Qdrant Cloud"""
    pass

@cli.command()
@click.argument('directory', type=click.Path(exists=True))
@click.option('--recreate', is_flag=True, help='Recreate the collection if it exists')
def upload(directory: str, recreate: bool):
    """Upload documents from a directory to Qdrant Cloud"""
    try:
        click.echo("Initializing uploader...")
        uploader = QdrantUploader()
        
        if recreate:
            click.echo("Recreating collection...")
            if not uploader.create_collection(recreate=True):
                click.echo("Failed to create collection")
                return

        click.echo(f"Processing documents from: {directory}")
        stats = uploader.process_directory(directory)
        
        click.echo("\nUpload Summary:")
        click.echo(f"Total files: {stats['total']}")
        click.echo(f"Successfully processed: {stats['success']}")
        click.echo(f"Failed: {stats['failed']}")
        
    except Exception as e:
        click.echo(f"Error: {str(e)}")

if __name__ == "__main__":
    cli()
