import os
import json
from pathlib import Path

# Updated imports to fix deprecation warnings
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    # Fallback to old import if new one not available
    from langchain.embeddings import HuggingFaceEmbeddings

try:
    from langchain_chroma import Chroma
except ImportError:
    # Fallback to old import if new one not available
    from langchain.vectorstores import Chroma

class ImageQueryDB:
    def __init__(self, db_path):
        """Initialize the database connection"""
        self.db_path = db_path
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database not found at {db_path}")
        
        self.db = Chroma(persist_directory=db_path, embedding_function=self.embeddings)
        print(f"Database loaded from: {db_path}")
    
    def clean_image_urls(self, urls):
        """Clean and validate image URLs"""
        if not urls:
            return []
        
        cleaned_urls = []
        for url in urls:
            if isinstance(url, str) and url.strip():
                # Remove all spaces and trailing dots
                cleaned_url = url.replace(' ', '').rstrip('.')
                if cleaned_url and cleaned_url.startswith('http'):
                    cleaned_urls.append(cleaned_url)
        
        return cleaned_urls
    
    def search_related_images(self, query_text, top_k=10, similarity_threshold=0.5):
        """
        Search for images related to the query text
        
        Args:
            query_text (str): The text to search for
            top_k (int): Number of top results to consider
            similarity_threshold (float): Minimum similarity score (0-1)
        
        Returns:
            dict: Search results with images and metadata
        """
        try:
            # Search for similar content
            results = self.db.similarity_search_with_score(query_text, k=top_k)
            
            if not results:
                return {
                    "query": query_text,
                    "total_results": 0,
                    "images_found": 0,
                    "results": []
                }
            
            # Process results and extract images
            processed_results = []
            all_images = []
            
            for doc, score in results:
                # Skip results with low similarity (higher score = lower similarity in some implementations)
                # Note: ChromaDB uses cosine distance, so lower scores are better
                if score > (1 - similarity_threshold):
                    continue
                
                # Extract image URLs from metadata
                image_urls_str = doc.metadata.get("image_urls", "")
                image_urls = []
                
                if image_urls_str:
                    raw_urls = image_urls_str.split('|')
                    image_urls = self.clean_image_urls(raw_urls)
                
                # Create result entry
                result_entry = {
                    "content": doc.page_content,
                    "similarity_score": round(1 - score, 4),  # Convert to similarity (higher = more similar)
                    "source": doc.metadata.get("source", "unknown"),
                    "document_title": doc.metadata.get("document_title", ""),
                    "section": doc.metadata.get("section", ""),
                    "content_type": doc.metadata.get("content_type", ""),
                    "document_type": doc.metadata.get("type", ""),
                    "has_images": len(image_urls) > 0,
                    "image_count": len(image_urls),
                    "image_urls": image_urls
                }
                
                processed_results.append(result_entry)
                all_images.extend(image_urls)
            
            # Remove duplicate images while preserving order
            unique_images = []
            seen_images = set()
            for img in all_images:
                if img not in seen_images:
                    unique_images.append(img)
                    seen_images.add(img)
            
            return {
                "query": query_text,
                "total_results": len(processed_results),
                "images_found": len(unique_images),
                "unique_images": unique_images,
                "results": processed_results
            }
            
        except Exception as e:
            print(f"Error searching database: {e}")
            return {
                "query": query_text,
                "error": str(e),
                "total_results": 0,
                "images_found": 0,
                "results": []
            }
    
    def find_images_by_text(self, query_text, return_only_images=True, top_k=5):
        """
        Simplified method to find images related to text
        
        Args:
            query_text (str): The text to search for
            return_only_images (bool): If True, returns only image URLs
            top_k (int): Number of results to consider
        
        Returns:
            list: List of image URLs or full results
        """
        search_results = self.search_related_images(query_text, top_k=top_k)
        
        if return_only_images:
            return search_results.get("unique_images", [])
        else:
            return search_results
    
    def print_search_results(self, query_text, top_k=5):
        """Print formatted search results"""
        print(f"\n{'='*80}")
        print(f"SEARCHING FOR: '{query_text}'")
        print('='*80)
        
        results = self.search_related_images(query_text, top_k=top_k)
        
        print(f"Total Results: {results['total_results']}")
        print(f"Images Found: {results['images_found']}")
        
        if results.get('error'):
            print(f"Error: {results['error']}")
            return
        
        if results['images_found'] > 0:
            print(f"\nUNIQUE IMAGE URLS:")
            print("-" * 40)
            for i, img_url in enumerate(results['unique_images'], 1):
                print(f"{i}. {img_url}")
        
        print(f"\nDETAILED RESULTS:")
        print("-" * 40)
        
        for i, result in enumerate(results['results'], 1):
            print(f"\nResult {i}:")
            print(f"  Source: {result['source']}")
            print(f"  Document: {result['document_title']}")
            print(f"  Section: {result['section']}")
            print(f"  Content Type: {result['content_type']}")
            print(f"  Similarity: {result['similarity_score']}")
            print(f"  Has Images: {result['has_images']} ({result['image_count']} images)")
            
            if result['image_urls']:
                print(f"  Image URLs:")
                for img_url in result['image_urls']:
                    print(f"    - {img_url}")
            
            print(f"  Content Preview: {result['content'][:200]}...")


def main():
    """Main function to demonstrate usage"""
    
    # Configuration
    db_path = "vector_db"  # Path to your ChromaDB database
    
    try:
        # Initialize the query system
        query_db = ImageQueryDB(db_path)
        
        # Test queries
        test_queries = [   "Download and Install the Barcode Client"]
        
        # Run test queries
        for query in test_queries:
            query_db.print_search_results(query, top_k=3)
            print("\n" + "="*80 + "\n")
        
        # Example of getting just image URLs
        print("EXAMPLE: Getting only image URLs for a specific query")
        print("-" * 60)
        
        sample_query = "Access Samsara Connector"
        image_urls = query_db.find_images_by_text(sample_query, return_only_images=True)
        
        print(f"Query: '{sample_query}'")
        print(f"Found {len(image_urls)} related images:")
        for i, url in enumerate(image_urls, 1):
            print(f"  {i}. {url}")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure you have created the ChromaDB database first using process_json.py")
    except Exception as e:
        print(f"Unexpected error: {e}")


def interactive_search():
    """Interactive search function"""
    db_path = "vector_db"
    
    try:
        query_db = ImageQueryDB(db_path)
        
        print("Interactive Image Search")
        print("Type 'quit' to exit")
        print("-" * 40)
        
        while True:
            query = input("\nEnter search text: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not query:
                print("Please enter some text to search for.")
                continue
            
            # Get images for the query
            images = query_db.find_images_by_text(query, return_only_images=True, top_k=5)
            
            if images:
                print(f"\nFound {len(images)} related images:")
                for i, img_url in enumerate(images, 1):
                    print(f"  {i}. {img_url}")
            else:
                print("No related images found.")
                
            # Ask if user wants detailed results
            show_details = input("\nShow detailed results? (y/n): ").strip().lower()
            if show_details in ['y', 'yes']:
                query_db.print_search_results(query, top_k=3)
                
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure you have created the ChromaDB database first using process_json.py")
    except KeyboardInterrupt:
        print("\nSearch interrupted. Goodbye!")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_search()
    else:
        main()