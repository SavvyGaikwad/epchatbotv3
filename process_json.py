import json
import os
import shutil
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

from langchain.docstore.document import Document
from langchain_community.vectorstores.utils import filter_complex_metadata

def load_json(filepath):
    """Load JSON file with error handling"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def clean_image_urls(image_paths):
    """Clean image URLs by removing unwanted spaces and fixing common issues"""
    if not image_paths:
        return []
    
    cleaned_urls = []
    for url in image_paths:
        if isinstance(url, str):
            # Remove spaces around dots and common URL issues
            cleaned_url = url.replace(' .', '.').replace('. ', '.').replace('  ', ' ').strip()
            # Remove spaces in github.com specifically
            cleaned_url = cleaned_url.replace('github. com', 'github.com')
            cleaned_url = cleaned_url.replace('github.  com', 'github.com')
            # Remove any remaining multiple spaces
            import re
            cleaned_url = re.sub(r'\s+', ' ', cleaned_url).strip()
            if cleaned_url:
                cleaned_urls.append(cleaned_url)
    
    return cleaned_urls

def safe_filter_metadata(metadata):
    """Safely filter metadata to only include simple types"""
    filtered = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            filtered[key] = value
        elif isinstance(value, list):
            # Convert list to string for storage
            filtered[key] = "|".join(str(item) for item in value if isinstance(item, (str, int, float, bool)))
        else:
            # Convert other types to string
            filtered[key] = str(value)
    return filtered

def process_content(data, source_file):
    """Process any JSON structure (handles steps, substeps, info, media, terms, questions, etc.)"""
    docs = []
    document_title = data.get("document_title", source_file)
    
    for section in data.get("content", []):
        section_title = section.get("title", "")
        section_content = []
        section_images = []
        current_step = None
        current_substeps = []
        
        for item in section.get("content", []):
            item_type = item.get("type", "")
            text = item.get("text", "")
            
            # Handle image paths for any content type
            item_images = []
            if "image_path" in item:
                raw_images = item["image_path"]
                if isinstance(raw_images, list):
                    item_images = clean_image_urls(raw_images)
                elif isinstance(raw_images, str):
                    item_images = clean_image_urls([raw_images])
                
                # Add to section images
                section_images.extend(item_images)
            
            # Handle different content types
            if item_type == "info" or item_type == "info.":  # Handle both formats
                content_text = f"Info: {text}"
                if item_images:
                    content_text += f"\nImages: {', '.join(item_images)}"
                section_content.append(content_text)
            
            elif item_type == "step":
                # Save previous step if exists
                if current_step and current_substeps:
                    full_step = f"{current_step}\nSubsteps: {' | '.join(current_substeps)}"
                    section_content.append(full_step)
                elif current_step:
                    section_content.append(current_step)
                
                # Start new step
                current_step = f"Step: {text}"
                if item_images:
                    current_step += f"\nImages: {', '.join(item_images)}"
                current_substeps = []
            
            elif item_type == "substep":
                if text:
                    substep_text = text
                    if item_images:
                        substep_text += f" [Images: {', '.join(item_images)}]"
                    current_substeps.append(substep_text)
            
            elif item_type == "media":
                image_path = item.get("path", "")
                if image_path:
                    # Clean the media path too
                    cleaned_path = clean_image_urls([image_path])
                    if cleaned_path:
                        section_images.extend(cleaned_path)
                # Also handle image_path in media items
                if item_images:
                    section_images.extend(item_images)
            
            elif item_type == "term":
                title = item.get("title", "")
                definition = item.get("definition", "")
                if title and definition:
                    content_text = f"Term: {title}\nDefinition: {definition}"
                    if item_images:
                        content_text += f"\nImages: {', '.join(item_images)}"
                    section_content.append(content_text)
            
            elif item_type == "question":
                content_text = f"Question: {text}"
                if item_images:
                    content_text += f"\nImages: {', '.join(item_images)}"
                section_content.append(content_text)
            
            elif item_type == "answer":
                answer_text = text
                links = item.get("links", [])
                if links:
                    answer_text += f"\nReferences: {', '.join(links)}"
                if item_images:
                    answer_text += f"\nImages: {', '.join(item_images)}"
                section_content.append(f"Answer: {answer_text}")
            
            # Handle any other text content
            elif text and item_type not in ["step", "substep", "info", "info.", "media", "term", "question", "answer"]:
                content_text = f"{item_type}: {text}"
                if item_images:
                    content_text += f"\nImages: {', '.join(item_images)}"
                section_content.append(content_text)
        
        # Don't forget the last step
        if current_step and current_substeps:
            full_step = f"{current_step}\nSubsteps: {' | '.join(current_substeps)}"
            section_content.append(full_step)
        elif current_step:
            section_content.append(current_step)
        
        # Create document if there's content
        if section_content or section_images:
            # Create comprehensive section document
            full_content = f"Document: {document_title}\nSection: {section_title}\n\n"
            if section_content:
                full_content += "\n".join(section_content)
            
            # Add all images found in the section
            if section_images:
                full_content += f"\n\nSection Images: {', '.join(section_images)}"
            
            images_str = "|".join(section_images) if section_images else ""
            
            metadata = {
                "source": source_file,
                "document_title": document_title,
                "section": section_title,
                "type": "section",
                "images": images_str,
                "has_images": len(section_images) > 0,
                "content_count": len(section_content),
                "image_count": len(section_images),
                "image_urls": "|".join(section_images)  # Convert list to string
            }
            
            # Apply safe filtering to metadata
            safe_metadata = safe_filter_metadata(metadata)
            docs.append(Document(page_content=full_content, metadata=safe_metadata))
            
            # Create individual documents for better searchability
            for i, content_item in enumerate(section_content):
                item_metadata = {
                    "source": source_file,
                    "document_title": document_title,
                    "section": section_title,
                    "item": i,
                    "type": "content_item",
                    "images": images_str,
                    "has_images": len(section_images) > 0,
                    "image_urls": "|".join(section_images)  # Convert list to string
                }
                
                # Apply safe filtering to metadata
                safe_item_metadata = safe_filter_metadata(item_metadata)
                docs.append(Document(
                    page_content=f"{document_title} - {section_title}: {content_item}",
                    metadata=safe_item_metadata
                ))
    
    return docs

def find_all_json_files(root_folder):
    """Recursively find all JSON files in the folder and subfolders"""
    json_files = []
    
    # Use pathlib for better path handling
    root_path = Path(root_folder)
    
    if not root_path.exists():
        print(f"Folder not found: {root_folder}")
        return []
    
    # Find all JSON files recursively
    for json_file in root_path.rglob("*.json"):
        json_files.append(str(json_file))
    
    return sorted(json_files)

def create_vector_db(data_folder, persist_dir):
    """Create vector database from all JSON files in data folder"""
    
    # Remove existing database
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
        print(f"Removed existing database at {persist_dir}")
    
    # Find all JSON files recursively
    json_files = find_all_json_files(data_folder)
    
    if not json_files:
        print(f"No JSON files found in {data_folder}")
        return
    
    print(f"Found {len(json_files)} JSON files:")
    for file in json_files:
        relative_path = os.path.relpath(file, data_folder)
        print(f"  - {relative_path}")
    
    all_docs = []
    file_stats = {}
    total_images = 0
    
    # Process each file
    for filepath in json_files:
        print(f"\nProcessing: {filepath}")
        
        data = load_json(filepath)
        if not data:
            continue
        
        source_file = os.path.basename(filepath)
        document_title = data.get("document_title", source_file)
        
        print(f"Document title: {document_title}")
        
        docs = process_content(data, source_file)
        all_docs.extend(docs)
        
        # Count images in this file
        file_images = sum(doc.metadata.get("image_count", 0) for doc in docs if doc.metadata.get("type") == "section")
        total_images += file_images
        
        # Track stats per file
        relative_path = os.path.relpath(filepath, data_folder)
        file_stats[relative_path] = {"docs": len(docs), "images": file_images}
        print(f"Created {len(docs)} documents with {file_images} images")
    
    print(f"\nTotal documents created: {len(all_docs)}")
    print(f"Total images found: {total_images}")
    
    if not all_docs:
        print("No documents to process!")
        return
    
    # Show file processing stats
    print("\nDocuments and images per file:")
    for file_path, stats in file_stats.items():
        print(f"  {file_path}: {stats['docs']} documents, {stats['images']} images")
    
    # Show document title distribution
    doc_titles = {}
    for doc in all_docs:
        doc_title = doc.metadata.get("document_title", doc.metadata.get("source", "unknown"))
        doc_titles[doc_title] = doc_titles.get(doc_title, 0) + 1
    
    print("\nDocument title distribution:")
    for doc_title, count in sorted(doc_titles.items()):
        print(f"  {doc_title}: {count} documents")
    
    # Create vector database
    print(f"\nCreating vector database...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Documents are already filtered with safe metadata, so we can use them directly
    # FIXED: Removed db.persist() call as it's no longer needed in newer Chroma versions
    db = Chroma.from_documents(all_docs, embeddings, persist_directory=persist_dir)
    print(f"Saved vector DB to {persist_dir}")
    
    return db

def test_db(db_path):
    """Test the database with various queries"""
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        return
    
    db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    
    # Test queries
    test_queries = [
        "cycle count",
        "saved view",
        "needs recount",
        "asset downtime report"
    ]
    
    print(f"\n{'='*80}")
    print("TESTING DATABASE")
    print('='*80)
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 50)
        
        results = db.similarity_search(query, k=3)
        
        if not results:
            print("  No results found")
            continue
        
        for i, result in enumerate(results, 1):
            source = result.metadata.get("source", "unknown")
            doc_title = result.metadata.get("document_title", "")
            section = result.metadata.get("section", "")
            
            print(f"  Result {i}: {source}")
            if doc_title and doc_title != source:
                print(f"    Document: {doc_title}")
            if section:
                print(f"    Section: {section}")
            print(f"    Content: {result.page_content[:150]}...")
            
            # Show images if available
            if result.metadata.get("has_images"):
                image_count = result.metadata.get("image_count", 0)
                image_urls = result.metadata.get("image_urls", "")
                if image_urls:
                    urls_list = image_urls.split('|')[:2]  # Show first 2 URLs
                    print(f"    Images: {image_count} image(s) - {', '.join(urls_list)}...")
            
            print()

def query_db(db_path, query, document_title=None, k=5):
    """Query the database with optional filtering"""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    
    # Basic similarity search
    results = db.similarity_search(query, k=k*2)  # Get more to filter
    
    # Apply filters if specified
    filtered_results = []
    for result in results:
        include = True
        
        if document_title and document_title.lower() not in result.metadata.get("document_title", "").lower():
            include = False
        
        if include:
            filtered_results.append(result)
        
        if len(filtered_results) >= k:
            break
    
    response_data = []
    
    for i, result in enumerate(filtered_results):
        # Convert images string back to list
        images_str = result.metadata.get("images", "")
        images_list = images_str.split('|') if images_str else []
        
        # Get cleaned image URLs from metadata (convert back from string to list)
        image_urls_str = result.metadata.get("image_urls", "")
        image_urls = image_urls_str.split('|') if image_urls_str else []
        
        item = {
            "rank": i + 1,
            "content": result.page_content,
            "source": result.metadata.get("source", "unknown"),
            "document_title": result.metadata.get("document_title", ""),
            "section": result.metadata.get("section", ""),
            "type": result.metadata.get("type", ""),
            "images": images_list,  # Backward compatibility
            "image_urls": image_urls,  # Clean URLs as list
            "has_images": result.metadata.get("has_images", False)
        }
        response_data.append(item)
    
    return response_data

if __name__ == "__main__":
    # Configuration
    data_folder = "data"  
    db_path = "vector_db"
    
    print("Creating comprehensive vector database...")
    
    # Create the database
    db = create_vector_db(data_folder, db_path)
    
    if db:
        print(f"\nDatabase created successfully!")
        # Test the database
        test_db(db_path)
    else:
        print("Failed to create database!")