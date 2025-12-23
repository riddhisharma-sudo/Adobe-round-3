import faiss
import numpy as np
import json
import os
import pickle
from typing import List, Dict, Any, Tuple
import uuid
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence_transformers not available, using TF-IDF fallback")

class VectorDatabase:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", index_file: str = "vector_index.faiss", metadata_file: str = "vector_metadata.json"):
        """
        Initialize the vector database with FAISS index and embedding model
        
        Args:
            model_name: Name of the sentence transformer model to use for embeddings
            index_file: Path to save/load the FAISS index
            metadata_file: Path to save/load the metadata (text content and document info)
        """
        self.use_sentence_transformers = SENTENCE_TRANSFORMERS_AVAILABLE
        
        if self.use_sentence_transformers:
            try:
                print("Loading sentence transformer model for CPU...")
                self.model = SentenceTransformer(model_name, device='cpu')
                self.dimension = self.model.get_sentence_embedding_dimension()
                self.model_name = model_name
                print(f"Sentence transformer loaded successfully with dimension {self.dimension} on CPU")
            except Exception as e:
                print(f"Error loading sentence transformer: {e}")
                print("Falling back to TF-IDF")
                self.use_sentence_transformers = False
        
        if not self.use_sentence_transformers:
            # Fallback to TF-IDF
            self.vectorizer = TfidfVectorizer(max_features=384, stop_words='english')
            self.dimension = 384
            self.texts_for_vectorizer = []
            self.model_name = "TF-IDF"
            print("Using TF-IDF vectorizer for embeddings")
        
        # Initialize FAISS index (using cosine similarity for better semantic search)
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product for cosine similarity
        
        # Metadata storage
        self.metadata = {}  # id -> {text, doc, page, heading, etc.}
        self.id_counter = 0
        
        # File paths
        self.index_file = index_file
        self.metadata_file = metadata_file
        
        # Load existing index if available
        self.load_index()
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate normalized embedding for a single text (for cosine similarity)"""
        if self.use_sentence_transformers:
            embedding = self.model.encode([text])[0]
            # Normalize for cosine similarity
            embedding = embedding / np.linalg.norm(embedding)
            return embedding
        else:
            # Use TF-IDF
            if hasattr(self, 'vectorizer') and hasattr(self.vectorizer, 'vocabulary_'):
                # Vectorizer is already fitted
                vector = self.vectorizer.transform([text]).toarray()[0]
            else:
                # Need to fit the vectorizer first with all texts
                if len(self.texts_for_vectorizer) == 0:
                    # If no texts available, return zero vector
                    return np.zeros(self.dimension, dtype=np.float32)
                
                all_texts = self.texts_for_vectorizer + [text]
                vectors = self.vectorizer.fit_transform(all_texts).toarray()
                vector = vectors[-1]  # Last vector is for the input text
            
            # Pad or truncate to match dimension
            if len(vector) < self.dimension:
                vector = np.pad(vector, (0, self.dimension - len(vector)))
            elif len(vector) > self.dimension:
                vector = vector[:self.dimension]
            
            # Normalize for cosine similarity
            vector = vector.astype(np.float32)
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            
            return vector
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate normalized embeddings for multiple texts (for cosine similarity)"""
        if self.use_sentence_transformers:
            embeddings = self.model.encode(texts)
            # Normalize all embeddings for cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            embeddings = embeddings / norms
            return embeddings
        else:
            # Use TF-IDF
            self.texts_for_vectorizer.extend(texts)
            vectors = self.vectorizer.fit_transform(texts).toarray()
            
            # Ensure all vectors have the same dimension and normalize
            processed_vectors = []
            for vector in vectors:
                if len(vector) < self.dimension:
                    vector = np.pad(vector, (0, self.dimension - len(vector)))
                elif len(vector) > self.dimension:
                    vector = vector[:self.dimension]
                
                # Normalize for cosine similarity
                vector = vector.astype(np.float32)
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm
                
                processed_vectors.append(vector)
            
            return np.array(processed_vectors, dtype=np.float32)
    
    def add_document_chunks(self, extracted_data: List[Dict[str, Any]]) -> List[str]:
        """
        Add chunks from extracted PDF data to the vector database
        
        Args:
            extracted_data: List of chunks from PDF extraction
            
        Returns:
            List of IDs assigned to the added chunks
        """
        if not extracted_data:
            return []
        
        texts = []
        chunk_metadata = []
        chunk_ids = []
        
        for chunk in extracted_data:
            # Create a unique ID for this chunk
            chunk_id = str(uuid.uuid4())
            chunk_ids.append(chunk_id)
            
            # Prepare text for embedding
            text = chunk.get('text', '')
            texts.append(text)
            
            # Store metadata
            metadata = {
                'id': chunk_id,
                'text': text,
                'doc': chunk.get('doc', ''),
                'page': chunk.get('page', 0),
                'heading': chunk.get('heading', ''),
                'original_chunk': chunk  # Store the original chunk data
            }
            
            chunk_metadata.append(metadata)
            self.metadata[chunk_id] = metadata
        
        # Generate embeddings
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.generate_embeddings(texts)
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Save the updated index and metadata
        self.save_index()
        
        print(f"Added {len(chunk_ids)} chunks to vector database")
        return chunk_ids
    
    def search_similar(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for semantically similar chunks using cosine similarity
        
        Args:
            query: Search query text
            top_k: Number of top similar results to return
            
        Returns:
            List of similar chunks with cosine similarity scores
        """
        if self.index.ntotal == 0:
            return []
        
        # Generate normalized query embedding for cosine similarity
        query_embedding = self.generate_embedding(query)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Search in FAISS index (using inner product for cosine similarity)
        similarities, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx >= 0:  # Valid index
                # Get metadata using index position (since we add in order)
                chunk_id = list(self.metadata.keys())[idx]
                metadata = self.metadata[chunk_id]
                
                # Similarity score is already cosine similarity (higher is better)
                similarity_score = float(similarity)
                
                result = {
                    'id': chunk_id,
                    'text': metadata['text'],
                    'doc': metadata['doc'],
                    'page': metadata['page'],
                    'heading': metadata['heading'],
                    'similarity_score': similarity_score,
                    'cosine_similarity': similarity_score,
                    'rank': i + 1,
                    'original_chunk': metadata['original_chunk']
                }
                results.append(result)
        
        return results
    
    def clear_database(self):
        """Clear all data from the vector database"""
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product for cosine similarity
        self.metadata = {}
        self.id_counter = 0
        self.save_index()
        print("Vector database cleared")
    
    def save_index(self):
        """Save the FAISS index and metadata to files"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, self.index_file)
            
            # Save metadata
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
            
            print(f"Vector database saved to {self.index_file} and {self.metadata_file}")
        except Exception as e:
            print(f"Error saving vector database: {e}")
    
    def load_index(self):
        """Load the FAISS index and metadata from files if they exist"""
        try:
            if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
                # Load FAISS index
                self.index = faiss.read_index(self.index_file)
                
                # Load metadata
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                
                print(f"Vector database loaded from {self.index_file} and {self.metadata_file}")
                print(f"Database contains {self.index.ntotal} vectors")
            else:
                print("No existing vector database found, starting fresh")
        except Exception as e:
            print(f"Error loading vector database: {e}")
            # Reset to empty state
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product for cosine similarity
            self.metadata = {}
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database"""
        docs = set()
        for metadata in self.metadata.values():
            docs.add(metadata['doc'])
        
        return {
            'total_chunks': len(self.metadata),
            'total_vectors': self.index.ntotal,
            'unique_documents': len(docs),
            'documents': list(docs),
            'embedding_dimension': self.dimension,
            'model_name': getattr(self, 'model_name', 'unknown'),
            'embedding_method': 'sentence_transformers' if self.use_sentence_transformers else 'tfidf',
            'similarity_metric': 'cosine_similarity',
            'device': 'cpu'
        }
    
    def get_processed_documents(self) -> List[str]:
        """Get a list of document names that are already processed in the vector database"""
        docs = set()
        for metadata in self.metadata.values():
            if metadata.get('doc'):
                docs.add(metadata['doc'])
        return list(docs)
    
    def is_document_processed(self, document_name: str) -> bool:
        """Check if a document is already processed in the vector database"""
        processed_docs = self.get_processed_documents()
        return document_name in processed_docs
    
    def remove_document(self, document_name: str) -> int:
        """
        Remove all chunks belonging to a specific document from the vector database.
        Returns the number of chunks removed.
        
        Note: This is a simplified implementation. In production, you might want to
        rebuild the entire FAISS index for better performance.
        """
        chunks_to_remove = []
        for chunk_id, metadata in self.metadata.items():
            if metadata.get('doc') == document_name:
                chunks_to_remove.append(chunk_id)
        
        # Remove from metadata
        for chunk_id in chunks_to_remove:
            del self.metadata[chunk_id]
        
        if chunks_to_remove:
            # For simplicity, rebuild the entire index
            # In production, you might want a more efficient approach
            self._rebuild_index()
            print(f"Removed {len(chunks_to_remove)} chunks for document: {document_name}")
        
        return len(chunks_to_remove)
    
    def _rebuild_index(self):
        """Rebuild the FAISS index from current metadata"""
        if not self.metadata:
            # Empty database
            self.index = faiss.IndexFlatIP(self.dimension)
            return
        
        # Extract texts and regenerate embeddings
        texts = [metadata['text'] for metadata in self.metadata.values()]
        embeddings = self.generate_embeddings(texts)
        
        # Create new index
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings.astype('float32'))
        
        # Save the rebuilt index
        self.save_index()
        print(f"Rebuilt FAISS index with {len(texts)} chunks")

# Global instance
vector_db = VectorDatabase(index_file=os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "vector_index.faiss"), metadata_file=os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "vector_metadata.json"))
