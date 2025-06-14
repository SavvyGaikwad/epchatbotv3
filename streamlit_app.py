# Alternative SQLite fix - add to top of streamlit_app.py
import subprocess
import sys

try:
    # Try to upgrade sqlite3 in the environment
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pysqlite3-binary"])
    import pysqlite3
    sys.modules['sqlite3'] = pysqlite3
except:
    pass  # Fall back to system sqlite3

import streamlit as st
import os
import sys
from pathlib import Path
import logging
from typing import List, Dict, Any
import time
from datetime import datetime
import requests
from PIL import Image
from io import BytesIO
import re
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.docstore.document import Document
    import google.generativeai as genai  # Added missing import
except ImportError as e:
    st.error(f"Missing dependencies: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Document Search Interface",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        border: none;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .chat-message {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .response-container {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 2px 20px rgba(0,0,0,0.08);
        margin: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .image-container {
        border: 1px solid #e1e5e9;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: #fafbfc;
    }
    
    .status-indicator {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    .status-success {
        background: #d4edda;
        color: #155724;
    }
    
    .status-info {
        background: #d1ecf1;
        color: #0c5460;
    }
    
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #ddd, transparent);
        margin: 2rem 0;
    }
    
    .language-selector {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class ProfessionalChatbot:
    def __init__(self):
        self.setup_genai()
        self.setup_vector_db()
        self.initialize_session_state()
        self.setup_language_prompts()
    
    def setup_genai(self):
        """Initialize Gemini AI model"""
        try:
            # Get API key from environment variable or use the hardcoded one
            api_key = os.getenv("GEMINI_API_KEY", "AIzaSyAvzloY_NyX-yjtZb8EE_RdXPs3rPmMEso")
            
            # Debug information
            logger.info(f"Using API key: {api_key[:10]}...{api_key[-4:] if len(api_key) > 14 else 'SHORT_KEY'}")
            
            genai.configure(api_key=api_key)
            
            # Test the API key by making a simple request
            self.model = genai.GenerativeModel(model_name="gemini-1.5-flash")
            
            # Test with a simple prompt to verify the API key works
            test_response = self.model.generate_content("Hello, are you working?")
            logger.info("Gemini AI model initialized and tested successfully")
            
        except Exception as e:
            st.error(f"Failed to initialize AI model. Error: {str(e)}")
            logger.error(f"Gemini AI initialization error: {e}")
            st.error("Please check your API key and ensure it's valid. You can get a new API key from: https://makersuite.google.com/app/apikey")
            st.stop()
    
    def setup_vector_db(self):
        """Initialize vector database"""
        try:
            self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            self.vector_db_path = "vector_db"
            
            if os.path.exists(self.vector_db_path):
                self.vector_db = Chroma(
                    persist_directory=self.vector_db_path, 
                    embedding_function=self.embedding_model
                )
               # status_text = "✅ Knowledge base connected" if st.session_state.get('language', 'English') == 'English' else "✅ ナレッジベースに接続しました"
               # st.markdown(
               #     f'<div class="status-indicator status-success">{status_text}</div>', 
               #     unsafe_allow_html=True
                #)
                logger.info(f"Vector database loaded from {self.vector_db_path}")
            else:
                error_text = "Knowledge base not found. Please contact system administrator." if st.session_state.get('language', 'English') == 'English' else "ナレッジベースが見つかりません。システム管理者にお問い合わせください。"
                st.error(error_text)
                logger.error(f"Vector database not found at {self.vector_db_path}")
                st.stop()
        except Exception as e:
            error_text = "Failed to connect to knowledge base." if st.session_state.get('language', 'English') == 'English' else "ナレッジベースへの接続に失敗しました。"
            st.error(error_text)
            logger.error(f"Vector database initialization error: {e}")
            st.stop()

    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'processing' not in st.session_state:
            st.session_state.processing = False
        if 'language' not in st.session_state:
            st.session_state.language = 'English'
        if 'query_submitted' not in st.session_state:
            st.session_state.query_submitted = False

    def setup_language_prompts(self):
        """Setup language-specific prompts and text"""
        self.language_config = {
            'English': {
                'title': 'Eptura Asset AI',
                'subtitle': 'Smartest AI answering all your questions for managing your company\'s Assets',
                'input_placeholder': 'Ask me anything about your documents:',
                'ask_button': '🔍 Ask Question',
                'clear_history': '🗑️ Clear Chat History',
                'recent_queries': '📝 Recent Queries',
                'response_header': '### Response',
                'visual_resources': '### Related Visual Resources',
                'quick_examples': '### 💡 Quick Start Examples',
                'documents_analyzed': 'Documents Analyzed',
                'content_types': 'Content Types',
                'visual_resources_metric': 'Visual Resources',
                'source_documents': 'Source Documents',
                'searching': 'Searching knowledge base...',
                'analyzing': 'Analyzing documents and generating response...',
                'no_docs_found': 'No relevant documents found. Please try rephrasing your question.',
                'enter_question': 'Please enter a question to get started.',
                'processing_error': 'An error occurred while processing your request. Please try again.',
                'examples': [
                ("📊 Data Management", "How to create and manage saved views?"),
                ("🔄 Process Workflows", "What are the cycle count procedures?"),
                ("👥 User Management", "How to manage user roles and permissions?"),
                ("⚙️ Configuration", "How to configure system settings?")
            ]
            
            },
            'Japanese': {
                'title': 'エプチュラ・アセット・AI',
                'subtitle': 'スマーテストAIがあなたのすべてのシツモンにコタエ、カイシャのアセットをカンリします',
                'input_placeholder': 'ドキュメントについて何でもお聞きください：',
                'ask_button': '🔍 質問する',
                'clear_history': '🗑️ 履歴をクリア',
                'recent_queries': '📝 最近の質問',
                'response_header': '### 回答',
                'visual_resources': '### 関連するビジュアルリソース',
                'quick_examples': '### 💡 クイックスタートの例',
                'documents_analyzed': '分析されたドキュメント',
                'content_types': 'コンテンツタイプ',
                'visual_resources_metric': 'ビジュアルリソース',
                'source_documents': 'ソースドキュメント',
                'searching': 'ナレッジベースを検索中...',
                'analyzing': 'ドキュメントを分析して回答を生成中...',
                'no_docs_found': '関連するドキュメントが見つかりませんでした。質問を言い換えてみてください。',
                'enter_question': '開始するには質問を入力してください。',
                'processing_error': 'リクエストの処理中にエラーが発生しました。再試行してください。',
                'examples': [
                ("📊 データ管理", "保存ビューの作成と管理方法は？"),
                ("🔄 プロセスワークフロー", "サイクルカウントの手順は何ですか？"),
                ("👥 ユーザー管理", "ユーザーの役割と権限の管理方法は？"),
                ("⚙️ 設定", "システム設定の構成方法は？")
            ]
            }
        }

    def get_text(self, key):
        """Get localized text based on current language"""
        return self.language_config[st.session_state.language][key]

    @staticmethod
    def convert_github_url_to_raw(github_url):
        """Convert GitHub URL to raw format and clean up common formatting issues"""
        # First, clean up the URL by removing trailing periods and extra spaces
        cleaned_url = github_url.strip().rstrip('.')
        # Remove any spaces in the URL (common issue with "github. com")
        cleaned_url = cleaned_url.replace(" ", "")
        
        if "github.com" in cleaned_url and "/blob/" in cleaned_url:
            # Replace github.com with raw.githubusercontent.com and remove /blob/
            raw_url = cleaned_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
            return raw_url
        elif "github.com" in cleaned_url and "/tree/" in cleaned_url:
            # Handle tree URLs as well
            raw_url = cleaned_url.replace("github.com", "raw.githubusercontent.com").replace("/tree/", "/")
            return raw_url
        return cleaned_url

    @staticmethod
    def is_gif_file(url):
        """Check if URL points to a GIF file"""
        return url.lower().endswith('.gif')

    def display_image_safely(self, image_url, container):
        """Display image with error handling and icon size detection"""
        try:
            raw_url = self.convert_github_url_to_raw(image_url)
            
            # Check if this is an icon
            is_icon = "icon" in raw_url.lower()
            
            with container:
                if self.is_gif_file(raw_url):
                    # For GIF files, use HTML with conditional sizing
                    if is_icon:
                        st.markdown(f"""
                        <div class="image-container">
                            <img src="{raw_url}" alt="Icon" 
                                style="height: 1em; width: auto; vertical-align: middle; border-radius: 2px;">
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="image-container">
                            <img src="{raw_url}" alt="Visual Resource" 
                                style="max-width: 100%; height: auto; border-radius: 4px;">
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    response = requests.get(raw_url, timeout=10)
                    response.raise_for_status()
                    img = Image.open(BytesIO(response.content))
                    
                    if is_icon:
                        # For icons, create a small inline display using HTML
                        import base64
                        from io import BytesIO as IOBytesIO
                        
                        # Convert image to base64 for inline display
                        buffered = IOBytesIO()
                        img.save(buffered, format="PNG")
                        img_str = base64.b64encode(buffered.getvalue()).decode()
                        
                        st.markdown(f"""
                        <div style="display: inline-block;">
                            <img src="data:image/png;base64,{img_str}" alt="Icon" 
                                style="height: 1em; width: auto; vertical-align: middle; border-radius: 2px;">
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.image(img, use_container_width=True)
            return True
        except Exception as e:
            logger.error(f"Image display error: {e}")
            with container:
                error_text = "Unable to load image" if st.session_state.language == 'English' else "画像を読み込めません"
                st.error(error_text)
            return False

    def extract_relevant_images_from_response(self, response_text, docs, max_images=3):
        """Extract images that are relevant to the generated response"""
        relevant_images = []
        seen_images = set()
        
        # Tokenize the response for matching
        response_tokens = set(response_text.lower().split())
        
        # Create a list of documents with their relevance scores
        doc_relevance_scores = []
        
        for doc in docs:
            # Calculate relevance score based on content overlap
            doc_content = doc.page_content.lower()
            doc_tokens = set(doc_content.split())
            
            # Calculate text overlap score
            overlap_score = len(response_tokens.intersection(doc_tokens)) / max(len(response_tokens), 1)
            
            # Get images from this document
            doc_images = []
            if hasattr(doc, 'metadata') and doc.metadata:
                images_str = doc.metadata.get('image_urls', '') or doc.metadata.get('images', '')
                
                if images_str and images_str.strip():
                    # Handle multiple delimiter formats
                    for delimiter in ['|', ',', ';']:
                        if delimiter in images_str:
                            doc_images = [img.strip() for img in images_str.split(delimiter) if img.strip()]
                            break
                    else:
                        doc_images = [images_str.strip()] if images_str.strip() else []
            
            # Store document with its relevance score and images
            if doc_images and overlap_score > 0:
                doc_relevance_scores.append({
                    'doc': doc,
                    'score': overlap_score,
                    'images': doc_images,
                    'content': doc_content
                })
        
        # Sort documents by relevance score (highest first)
        doc_relevance_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Extract images from most relevant documents first
        for doc_info in doc_relevance_scores:
            if len(relevant_images) >= max_images:
                break
                
            for img_url in doc_info['images']:
                if img_url not in seen_images and len(relevant_images) < max_images:
                    relevant_images.append({
                        'url': img_url,
                        'relevance_score': doc_info['score'],
                        'source_doc': doc_info['doc'].metadata.get('document_title', 'Unknown'),
                        'section': doc_info['doc'].metadata.get('section', 'Unknown')
                    })
                    seen_images.add(img_url)
        
        return relevant_images

    def extract_images_with_content_matching(self, response_text, docs, max_images=3):
        """Advanced image extraction based on content relevance and semantic matching"""
        import re
        from collections import defaultdict
        
        relevant_images = []
        seen_images = set()
        
        # Extract key phrases from response (noun phrases, important terms)
        response_lower = response_text.lower()
        
        # Simple keyword extraction (you can enhance this with NLP libraries)
        # Remove common stop words and extract meaningful terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}
        
        # Extract words longer than 3 characters that aren't stop words
        response_keywords = set()
        for word in re.findall(r'\b\w+\b', response_lower):
            if len(word) > 3 and word not in stop_words:
                response_keywords.add(word)
        
        # Score each document based on multiple factors
        doc_scores = []
        
        for doc in docs:
            if not hasattr(doc, 'metadata') or not doc.metadata:
                continue
                
            # Get images from document
            images_str = doc.metadata.get('image_urls', '') or doc.metadata.get('images', '')
            if not images_str or not images_str.strip():
                continue
                
            # Parse images
            doc_images = []
            for delimiter in ['|', ',', ';']:
                if delimiter in images_str:
                    doc_images = [img.strip() for img in images_str.split(delimiter) if img.strip()]
                    break
            else:
                doc_images = [images_str.strip()] if images_str.strip() else []
            
            if not doc_images:
                continue
            
            # Calculate relevance score
            doc_content_lower = doc.page_content.lower()
            doc_keywords = set(re.findall(r'\b\w+\b', doc_content_lower))
            
            # Keyword overlap score
            keyword_overlap = len(response_keywords.intersection(doc_keywords))
            keyword_score = keyword_overlap / max(len(response_keywords), 1)
            
            # Content type bonus (prioritize certain types)
            content_type = doc.metadata.get('content_type', '').lower()
            type_bonus = 0
            if any(term in content_type for term in ['image', 'figure', 'diagram', 'chart', 'visual']):
                type_bonus = 0.2
            
            # Section relevance (if response mentions section topics)
            section_bonus = 0
            section_title = doc.metadata.get('section', '').lower()
            if section_title and any(keyword in section_title for keyword in response_keywords):
                section_bonus = 0.1
            
            # Final score
            final_score = keyword_score + type_bonus + section_bonus
            
            if final_score > 0:  # Only include documents with some relevance
                doc_scores.append({
                    'doc': doc,
                    'images': doc_images,
                    'score': final_score,
                    'keyword_matches': keyword_overlap
                })
        
        # Sort by relevance score
        doc_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Extract images from most relevant documents
        for doc_info in doc_scores:
            if len(relevant_images) >= max_images:
                break
                
            for img_url in doc_info['images']:
                if img_url not in seen_images and len(relevant_images) < max_images:
                    relevant_images.append({
                        'url': img_url,
                        'relevance_score': doc_info['score'],
                        'keyword_matches': doc_info['keyword_matches'],
                        'source_doc': doc_info['doc'].metadata.get('document_title', 'Unknown'),
                        'section': doc_info['doc'].metadata.get('section', 'Unknown'),
                        'content_type': doc_info['doc'].metadata.get('content_type', 'Unknown')
                    })
                    seen_images.add(img_url)
        
        return relevant_images

    def generate_response(self, query, docs):
        """Generate AI response from documents"""
        # Build context from documents
        context_parts = []
        for i, doc in enumerate(docs):
            metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            source = metadata.get('source', 'Document')
            doc_title = metadata.get('document_title', source)
            section = metadata.get('section', '')
            doc_type = metadata.get('document_type', 'Content')
            
            header = f"[Source {i+1}: {doc_title}"
            if section:
                header += f" - {section}"
            header += f" ({doc_type})]"
            
            context_parts.append(f"{header}\n{doc.page_content}")
        
        context = "\n\n---\n\n".join(context_parts)

        # Create language-specific prompt
        if st.session_state.language == 'Japanese':
            prompt = f"""
プロフェッショナルなドキュメントアシスタントとして、提供されたコンテキストに基づいて包括的で正確な回答を日本語で提供してください。

ガイドライン:
- 明確でプロフェッショナルな日本語を使用する
- 回答を論理的に構成し、適切な書式設定を行う
- 回答に「(ソース1、2、3)」のようなソース引用や参照を含めない
- 情報を直接的で権威ある知識として提示する
- 情報が不完全な場合は制限を認める
- 可能な場合は実用的なガイダンスを提供する
- ステップバイステップの指示には箇条書きや番号付きリストを使用する
- 会話的でありながらプロフェッショナルな回答を維持する

ナレッジベースからのコンテキスト:
{context}

ユーザーの質問: {query}

ソース引用なしで詳細なプロフェッショナルな回答を日本語で提供してください:
"""
        else:
            prompt = f"""
As a professional document assistant, provide a comprehensive and accurate answer based on the provided context.

Guidelines:
- Use clear, professional language
- Structure your response logically with proper formatting
- Do NOT include source citations or references like "(Source 1, 2, 3)" in your response
- Present information as direct, authoritative knowledge
- If information is incomplete, acknowledge limitations
- Provide actionable guidance when possible
- Use bullet points or numbered lists for step-by-step instructions
- Keep responses conversational yet professional

Context from knowledge base:
{context}

User Question: {query}

Please provide a detailed, professional response without any source citations:
"""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"AI response generation error: {e}")
            error_text = "I apologize, but I'm unable to generate a response at this time. Please try again or contact support." if st.session_state.language == 'English' else "申し訳ございませんが、現在回答を生成できません。再試行するかサポートにお問い合わせください。"
            return error_text
    
    def display_inline_media(self, content_text, images_list):
        """
        Display content with inline media similar to document format
        Regular images are displayed below the text they relate to
        Icons are displayed next to (inline with) the specific line they relate to
        """
        if not images_list:
            st.write(content_text)
            return
        
        # Separate icons from regular images
        icons = [img for img in images_list if "icon" in img['url'].lower()]
        regular_images = [img for img in images_list if "icon" not in img['url'].lower()]
        
        # Split content into lines for icon placement and paragraphs for regular images
        lines = content_text.split('\n')
        paragraphs = content_text.split('\n\n')
        
        # If we have icons, try to place them next to relevant lines
        if icons:
            # Distribute icons across lines
            lines_with_content = [line for line in lines if line.strip()]
            if lines_with_content:
                icons_per_line = len(icons) // max(len(lines_with_content), 1)
                remaining_icons = len(icons) % max(len(lines_with_content), 1)
                
                icon_index = 0
                processed_lines = []
                
                for line_idx, line in enumerate(lines):
                    if line.strip() and icon_index < len(icons):
                        # Calculate how many icons for this line
                        icons_for_this_line = icons_per_line
                        if line_idx < remaining_icons:
                            icons_for_this_line += 1
                        
                        # Create icon HTML for this line
                        line_icon_html = ""
                        for _ in range(icons_for_this_line):
                            if icon_index < len(icons):
                                icon_info = icons[icon_index]
                                try:
                                    raw_url = self.convert_github_url_to_raw(icon_info['url'])
                                    line_icon_html += f' <img src="{raw_url}" alt="Icon" style="height: 1em; width: auto; vertical-align: middle; margin: 0 2px; border-radius: 2px;">'
                                except:
                                    pass
                                icon_index += 1
                        
                        # Add line with inline icons
                        if line_icon_html:
                            processed_lines.append(f"{line} {line_icon_html}")
                        else:
                            processed_lines.append(line)
                    else:
                        processed_lines.append(line)
                
                # Display content with inline icons
                st.markdown('\n'.join(processed_lines), unsafe_allow_html=True)
            else:
                st.write(content_text)
        else:
            # No icons, just display paragraphs normally for regular image processing
            pass
        
        # Handle regular images - display below paragraphs
        if regular_images:
            if not icons:  # If no icons were processed, we need to display the text first
                paragraphs_to_process = paragraphs
            else:
                # Text already displayed with icons, now just handle regular images
                paragraphs_to_process = paragraphs
            
            images_per_paragraph = len(regular_images) // max(len(paragraphs_to_process), 1)
            remaining_images_count = len(regular_images) % max(len(paragraphs_to_process), 1)
            
            image_index = 0
            
            # If icons were already displayed, skip text display and just show images
            if not icons:
                for para_idx, paragraph in enumerate(paragraphs_to_process):
                    if paragraph.strip():
                        # Display the paragraph
                        st.write(paragraph)
                        
                        # Calculate how many images to show below this paragraph
                        images_to_show = images_per_paragraph
                        if para_idx < remaining_images_count:
                            images_to_show += 1
                        
                        # Display regular images below this paragraph
                        for _ in range(images_to_show):
                            if image_index < len(regular_images):
                                image_info = regular_images[image_index]
                                
                                # Create a container for the image with some spacing
                                st.markdown("<br>", unsafe_allow_html=True)
                                
                                # Display image with caption
                                image_container = st.container()
                                success = self.display_image_safely(image_info['url'], image_container)
                                
                                if success:
                                    # Add image caption/metadata if available
                                    caption_parts = []
                                    if 'source_doc' in image_info:
                                        caption_parts.append(f"Source: {image_info['source_doc']}")
                                    if 'section' in image_info:
                                        caption_parts.append(f"Section: {image_info['section']}")
                                    
                                    if caption_parts:
                                        st.caption(" | ".join(caption_parts))
                                
                                st.markdown("<br>", unsafe_allow_html=True)
                                image_index += 1
            else:
                # Icons already displayed with text, just show regular images distributed
                st.markdown("<br>", unsafe_allow_html=True)
                for image_info in regular_images:
                    image_container = st.container()
                    success = self.display_image_safely(image_info['url'], image_container)
                    
                    if success:
                        caption_parts = []
                        if 'source_doc' in image_info:
                            caption_parts.append(f"Source: {image_info['source_doc']}")
                        if 'section' in image_info:
                            caption_parts.append(f"Section: {image_info['section']}")
                        
                        if caption_parts:
                            st.caption(" | ".join(caption_parts))
                    
                    st.markdown("<br>", unsafe_allow_html=True)
        elif not icons:
            # No images at all, just display content normally
            st.write(content_text)

    def display_content_with_contextual_media(self, content_text, images_list):
        """
        Advanced inline media display that tries to match images to specific content sections
        Only displays images with relevance score >= 0.3
        Icons are placed next to the specific line they relate to
        Regular images are placed below the section they relate to
        """
        if not images_list:
            st.write(content_text)
            return
        
        # Additional filtering at display level for extra safety
        high_relevance_images = [
            img for img in images_list 
            if img.get('relevance_score', 0) >= 0.3
        ]
        
        if not high_relevance_images:
            st.write(content_text)
            return
        
        # Separate icons from regular images
        icons = [img for img in high_relevance_images if "icon" in img['url'].lower()]
        regular_images = [img for img in high_relevance_images if "icon" not in img['url'].lower()]
        
        # Split content into logical sections (paragraphs, lists, etc.)
        sections = self.split_content_into_sections(content_text)
        
        # Match regular images to sections based on content relevance
        section_image_mapping = self.match_images_to_sections(sections, regular_images)
        
        # Match icons to specific lines within sections
        section_icon_mapping = self.match_icons_to_lines(sections, icons)
        
        # Display content with inline media
        for section_idx, section in enumerate(sections):
            section_content = section['content']
            
            # Check if this section has icons that need inline placement
            if section_idx in section_icon_mapping:
                # Split section content into lines for icon placement
                lines = section_content.split('\n')
                section_icons = section_icon_mapping[section_idx]
                
                # Distribute icons across lines in this section
                lines_with_content = [line for line in lines if line.strip()]
                if lines_with_content and section_icons:
                    icons_per_line = len(section_icons) // max(len(lines_with_content), 1)
                    remaining_icons = len(section_icons) % max(len(lines_with_content), 1)
                    
                    icon_index = 0
                    processed_lines = []
                    
                    for line_idx, line in enumerate(lines):
                        if line.strip() and icon_index < len(section_icons):
                            # Calculate how many icons for this line
                            icons_for_this_line = icons_per_line
                            if line_idx < remaining_icons:
                                icons_for_this_line += 1
                            
                            # Create icon HTML for this line
                            line_icon_html = ""
                            for _ in range(icons_for_this_line):
                                if icon_index < len(section_icons):
                                    icon_info = section_icons[icon_index]
                                    try:
                                        raw_url = self.convert_github_url_to_raw(icon_info['url'])
                                        line_icon_html += f' <img src="{raw_url}" alt="Icon" style="height: 1em; width: auto; vertical-align: middle; margin: 0 2px; border-radius: 2px;">'
                                    except:
                                        pass
                                    icon_index += 1
                            
                            # Add line with inline icons
                            if line_icon_html:
                                processed_lines.append(f"{line} {line_icon_html}")
                            else:
                                processed_lines.append(line)
                        else:
                            processed_lines.append(line)
                    
                    # Display section with inline icons based on section type
                    processed_content = '\n'.join(processed_lines)
                    if section['type'] == 'paragraph':
                        st.markdown(processed_content, unsafe_allow_html=True)
                    elif section['type'] == 'list':
                        st.markdown(processed_content, unsafe_allow_html=True)
                    elif section['type'] == 'header':
                        st.markdown(f"### {processed_content}", unsafe_allow_html=True)
                    else:
                        st.markdown(processed_content, unsafe_allow_html=True)
                else:
                    # No icons for this section, display normally
                    if section['type'] == 'paragraph':
                        st.write(section_content)
                    elif section['type'] == 'list':
                        st.markdown(section_content)
                    elif section['type'] == 'header':
                        st.markdown(f"### {section_content}")
                    else:
                        st.write(section_content)
            else:
                # No icons for this section, display normally
                if section['type'] == 'paragraph':
                    st.write(section_content)
                elif section['type'] == 'list':
                    st.markdown(section_content)
                elif section['type'] == 'header':
                    st.markdown(f"### {section_content}")
                else:
                    st.write(section_content)
            
            # Display associated regular images below this section (only those with relevance >= 0.3)
            if section_idx in section_image_mapping:
                for image_info in section_image_mapping[section_idx]:
                    relevance_score = image_info.get('relevance_score', 0)
                    
                    # Double-check relevance before displaying
                    if relevance_score >= 0.3:
                        st.markdown("<div style='margin: 10px 0;'>", unsafe_allow_html=True)
                        image_container = st.container()
                        success = self.display_image_safely(image_info['url'], image_container)
                        st.markdown("</div>", unsafe_allow_html=True)

    def match_icons_to_lines(self, sections, icons_list):
        """Match icons to specific sections for line-level placement"""
        if not icons_list:
            return {}
        
        section_icon_mapping = {}
        
        # Simple distribution strategy for icons across sections
        icons_per_section = len(icons_list) // max(len(sections), 1)
        remaining_icons = len(icons_list) % max(len(sections), 1)
        
        icon_index = 0
        
        for section_idx, section in enumerate(sections):
            # Skip headers for icon placement
            if section['type'] == 'header':
                continue
            
            icons_for_this_section = icons_per_section
            if section_idx < remaining_icons:
                icons_for_this_section += 1
            
            section_icons = []
            for _ in range(icons_for_this_section):
                if icon_index < len(icons_list):
                    section_icons.append(icons_list[icon_index])
                    icon_index += 1
            
            if section_icons:
                section_icon_mapping[section_idx] = section_icons
        
        return section_icon_mapping

    def split_content_into_sections(self, content_text):
        """Split content into logical sections for better media placement"""
        sections = []
        lines = content_text.split('\n')
        current_section = ""
        current_type = "paragraph"
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_section:
                    sections.append({
                        'content': current_section.strip(),
                        'type': current_type
                    })
                    current_section = ""
                continue
            
            # Detect headers
            if line.startswith('#') or (len(line) < 100 and line.isupper()):
                if current_section:
                    sections.append({
                        'content': current_section.strip(),
                        'type': current_type
                    })
                sections.append({
                    'content': line.replace('#', '').strip(),
                    'type': 'header'
                })
                current_section = ""
                current_type = "paragraph"
            # Detect lists
            elif line.startswith(('-', '*', '•')) or re.match(r'^\d+\.', line):
                if current_type != "list":
                    if current_section:
                        sections.append({
                            'content': current_section.strip(),
                            'type': current_type
                        })
                        current_section = ""
                    current_type = "list"
                current_section += f"\n{line}" if current_section else line
            else:
                if current_type != "paragraph":
                    if current_section:
                        sections.append({
                            'content': current_section.strip(),
                            'type': current_type
                        })
                        current_section = ""
                    current_type = "paragraph"
                current_section += f"\n{line}" if current_section else line
        
        # Add final section
        if current_section:
            sections.append({
                'content': current_section.strip(),
                'type': current_type
            })
        
        return sections

    def match_images_to_sections(self, sections, images_list):
        """Match images to content sections based on relevance"""
        section_image_mapping = {}
        
        if not images_list:
            return section_image_mapping
        
        # Simple distribution strategy - can be enhanced with NLP
        images_per_section = len(images_list) // max(len(sections), 1)
        remaining_images = len(images_list) % max(len(sections), 1)
        
        image_index = 0
        
        for section_idx, section in enumerate(sections):
            # Skip headers for image placement
            if section['type'] == 'header':
                continue
            
            images_for_this_section = images_per_section
            if section_idx < remaining_images:
                images_for_this_section += 1
            
            section_images = []
            for _ in range(images_for_this_section):
                if image_index < len(images_list):
                    section_images.append(images_list[image_index])
                    image_index += 1
            
            if section_images:
                section_image_mapping[section_idx] = section_images
        
        return section_image_mapping

    def process_query(self, query, container=None):
        """Updated process_query to store images in chat history"""
        try:
            # Use provided container or create a new one
            if container is None:
                container = st.container()
            
            with container:
                # Search documents
                with st.spinner(self.get_text('searching')):
                    docs = self.vector_db.similarity_search(query, k=10)
                
                if not docs:
                    st.warning(self.get_text('no_docs_found'))
                    return
                
                # Generate response
                with st.spinner(self.get_text('analyzing')):
                    response_text = self.generate_response(query, docs)
                
                # Extract relevant images based on response content
                all_relevant_images = self.extract_images_with_content_matching(
                    response_text, docs, max_images=10  # Get more initially
                )
                
                # Filter images by relevance threshold (0.3)
                filtered_images = [
                    img for img in all_relevant_images 
                    if img.get('relevance_score', 0) >= 0.3
                ]
                
                # Limit to 5 images after filtering
                relevant_images = filtered_images[:5]
                
                # Display response with inline media
                st.markdown('<div class="response-container">', unsafe_allow_html=True)
                st.markdown(self.get_text('response_header'))
                
                # Use inline media display instead of separate sections
                if relevant_images:
                    self.display_content_with_contextual_media(response_text, relevant_images)
                else:
                    self.display_inline_media(response_text, [])
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Add to chat history with images
                timestamp = datetime.now().strftime("%H:%M")
                st.session_state.chat_history.append({
                    'timestamp': timestamp,
                    'query': query,
                    'response': response_text,
                    'doc_count': len(docs),
                    'image_count': len(relevant_images),
                    'images': relevant_images  # Store images for history display
                })
            
        except Exception as e:
            logger.error(f"Query processing error: {e}")
            if container:
                with container:
                    st.error(self.get_text('processing_error'))
            else:
                st.error(self.get_text('processing_error'))

    def display_response_with_inline_media_simple(self, response_text, images_list):
        """
        Simplified version - displays regular images below every major paragraph break
        Icons are displayed next to the specific line they relate to within paragraphs
        Similar to document format where regular images appear below relevant text
        """
        if not images_list:
            st.write(response_text)
            return
        
        # Separate icons from regular images
        icons = [img for img in images_list if "icon" in img['url'].lower()]
        regular_images = [img for img in images_list if "icon" not in img['url'].lower()]
        
        # Split by double newlines (paragraph breaks)
        paragraphs = response_text.split('\n\n')
        
        # Calculate distribution for regular images only
        total_paragraphs = len([p for p in paragraphs if p.strip()])
        if total_paragraphs == 0:
            st.write(response_text)
            return
        
        images_per_para = max(1, len(regular_images) // total_paragraphs) if regular_images else 0
        
        # Distribute icons across paragraphs for line-level placement
        icons_per_para = max(1, len(icons) // total_paragraphs) if icons else 0
        
        para_count = 0
        image_index = 0
        icon_index = 0
        
        for paragraph in paragraphs:
            if paragraph.strip():
                para_count += 1
                
                # Handle icons - place them next to lines within this paragraph
                icons_for_this_para = icons_per_para
                if para_count <= len(icons) % total_paragraphs:
                    icons_for_this_para += 1
                    
                current_paragraph_icons = []
                for _ in range(icons_for_this_para):
                    if icon_index < len(icons):
                        current_paragraph_icons.append(icons[icon_index])
                        icon_index += 1
                
                # Process paragraph with inline icons
                if current_paragraph_icons:
                    # Split paragraph into lines for icon placement
                    lines = paragraph.split('\n')
                    lines_with_content = [line for line in lines if line.strip()]
                    
                    if lines_with_content:
                        # Distribute icons across lines in this paragraph
                        icons_per_line = len(current_paragraph_icons) // max(len(lines_with_content), 1)
                        remaining_para_icons = len(current_paragraph_icons) % max(len(lines_with_content), 1)
                        
                        para_icon_index = 0
                        processed_lines = []
                        
                        for line_idx, line in enumerate(lines):
                            if line.strip() and para_icon_index < len(current_paragraph_icons):
                                # Calculate how many icons for this line
                                icons_for_this_line = icons_per_line
                                if line_idx < remaining_para_icons:
                                    icons_for_this_line += 1
                                
                                # Create icon HTML for this line
                                line_icon_html = ""
                                for _ in range(icons_for_this_line):
                                    if para_icon_index < len(current_paragraph_icons):
                                        icon_info = current_paragraph_icons[para_icon_index]
                                        try:
                                            raw_url = self.convert_github_url_to_raw(icon_info['url'])
                                            line_icon_html += f' <img src="{raw_url}" alt="Icon" style="height: 1em; width: auto; vertical-align: middle; margin: 0 2px; border-radius: 2px;">'
                                        except:
                                            pass
                                        para_icon_index += 1
                                
                                # Add line with inline icons
                                if line_icon_html:
                                    processed_lines.append(f"{line} {line_icon_html}")
                                else:
                                    processed_lines.append(line)
                            else:
                                processed_lines.append(line)
                        
                        # Display paragraph with inline icons
                        st.markdown('\n'.join(processed_lines), unsafe_allow_html=True)
                    else:
                        # No lines with content, display paragraph normally
                        st.write(paragraph)
                else:
                    # No icons for this paragraph, display normally
                    st.write(paragraph)
                
                # Show regular images below this paragraph if we have any left
                if regular_images:
                    images_to_show = min(images_per_para, len(regular_images) - image_index)
                    
                    # For the last paragraph, show all remaining regular images
                    if para_count == total_paragraphs:
                        images_to_show = len(regular_images) - image_index
                    
                    for _ in range(images_to_show):
                        if image_index < len(regular_images):
                            image_info = regular_images[image_index]
                            
                            # Add some spacing
                            st.markdown("---")
                            
                            # Display image below the paragraph
                            image_container = st.container()
                            success = self.display_image_safely(image_info['url'], image_container)
                            
                            if success:
                                # Simple caption
                                caption_text = f"**Source:** {image_info.get('source_doc', 'Unknown')}"
                                if 'section' in image_info and image_info['section'] != 'Unknown':
                                    caption_text += f" | **Section:** {image_info['section']}"
                                st.caption(caption_text)
                            
                            image_index += 1
                            st.markdown("---")
            else:
                st.write(paragraph)  # Empty paragraphs for spacing

    def enhanced_process_json_content(data, source_file):
        """Enhanced JSON processing with better image-text relationship tracking"""
        docs = []
        document_title = data.get("document_title", source_file.replace('.json', ''))
        
        # Process each section
        for section in data.get("content", []):
            section_title = section.get("title", "").rstrip('.')
            section_type = section.get("type", "").rstrip('.')
            section_content_items = []
            all_section_images = []
            
            # Process content items in the section
            for item in section.get("content", []):
                # Extract text content
                content_text = extract_content_text(item)
                if content_text:
                    section_content_items.append(content_text)
                
                # Extract and clean image URLs
                item_images = []
                if "image_path" in item:
                    raw_images = item["image_path"]
                    if isinstance(raw_images, list):
                        item_images = clean_image_urls(raw_images)
                    elif isinstance(raw_images, str):
                        item_images = clean_image_urls([raw_images])
                    all_section_images.extend(item_images)
                
                # Create individual document for each content item with enhanced metadata
                if content_text:
                    # Create content fingerprint for better matching
                    content_keywords = set(re.findall(r'\b\w+\b', content_text.lower()))
                    content_keywords = {word for word in content_keywords if len(word) > 3}
                    
                    item_metadata = {
                        "source": source_file,
                        "document_title": document_title,
                        "section": section_title,
                        "section_type": section_type,
                        "content_type": item.get("type", "").rstrip('.'),
                        "type": "content_item",
                        "has_images": len(item_images) > 0,
                        "image_count": len(item_images),
                        "image_urls": "|".join(item_images) if item_images else "",
                        "content_keywords": "|".join(list(content_keywords)[:20]),  # Store top keywords
                        "content_length": len(content_text)
                    }
                    
                    # Enhanced content with image context
                    full_content = content_text
                    if item_images:
                        # Add image context to make matching more effective
                        full_content += f"\n\nRelated Images: {', '.join(item_images)}"
                        full_content += f"\nThis content has {len(item_images)} associated image(s)."
                    
                    safe_metadata = safe_filter_metadata(item_metadata)
                    docs.append(Document(
                        page_content=full_content,
                        metadata=safe_metadata
                    ))
            
            # Enhanced section-level document
            if section_content_items:
                section_keywords = set()
                for content in section_content_items:
                    content_words = set(re.findall(r'\b\w+\b', content.lower()))
                    section_keywords.update({word for word in content_words if len(word) > 3})
                
                section_full_content = f"Document: {document_title}\nSection: {section_title}\n\n"
                section_full_content += "\n\n".join(section_content_items)
                
                if all_section_images:
                    section_full_content += f"\n\nSection Images: {', '.join(all_section_images)}"
                    section_full_content += f"\nThis section contains {len(all_section_images)} images related to the content."
                
                section_metadata = {
                    "source": source_file,
                    "document_title": document_title,
                    "section": section_title,
                    "section_type": section_type,
                    "type": "section",
                    "has_images": len(all_section_images) > 0,
                    "image_count": len(all_section_images),
                    "content_items_count": len(section_content_items),
                    "image_urls": "|".join(all_section_images) if all_section_images else "",
                    "section_keywords": "|".join(list(section_keywords)[:30])  # Store more keywords for sections
                }
                
                safe_section_metadata = safe_filter_metadata(section_metadata)
                docs.append(Document(
                    page_content=section_full_content,
                    metadata=safe_section_metadata
                ))
        
        return docs

    def display_chat_history_conversation(self, chat_item):
        """Display a specific conversation from chat history"""
        st.markdown("---")
        
        # Display query
        query_label = "**Your Question:**" if st.session_state.language == 'English' else "**あなたの質問:**"
        st.markdown(f"{query_label}")
        st.markdown(f'<div style="background-color: #f0f2f6; padding: 10px; border-radius: 8px; margin-bottom: 15px;">{chat_item["query"]}</div>', unsafe_allow_html=True)
        
        # Display response
        response_label = "**AI Response:**" if st.session_state.language == 'English' else "**AI回答:**"
        st.markdown(f"{response_label}")
        
        # Display the response with images if available
        if 'images' in chat_item and chat_item['images']:
            self.display_content_with_contextual_media(chat_item['response'], chat_item['images'])
        else:
            st.markdown(f'<div style="background-color: #ffffff; padding: 15px; border-left: 4px solid #667eea; border-radius: 8px;">{chat_item["response"]}</div>', unsafe_allow_html=True)
        
        # Display metadata
        metadata_label = "**Details:**" if st.session_state.language == 'English' else "**詳細:**"
        st.markdown(f"{metadata_label}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            docs_label = "Documents:" if st.session_state.language == 'English' else "ドキュメント:"
            st.metric(docs_label, chat_item.get('doc_count', 0))
        with col2:
            images_label = "Images:" if st.session_state.language == 'English' else "画像:"
            st.metric(images_label, chat_item.get('image_count', 0))
        with col3:
            time_label = "Time:" if st.session_state.language == 'English' else "時刻:"
            st.metric(time_label, chat_item.get('timestamp', 'Unknown'))

    def render_sidebar(self):
        """Updated render sidebar with enhanced chat history"""
        with st.sidebar:
            # Language Selection at the top
            st.markdown("""
            <div class="language-selector">
                <h3>🌐 Language / 言語</h3>
            </div>
            """, unsafe_allow_html=True)
            
            language_options = {'English': '🇺🇸 English', 'Japanese': '🇯🇵 日本語'}
            
            new_language = st.selectbox(
                "",
                options=list(language_options.keys()),
                format_func=lambda x: language_options[x],
                index=0 if st.session_state.language == 'English' else 1,
                key="language_selector"
            )
            
            if new_language != st.session_state.language:
                st.session_state.language = new_language
                st.rerun()
            
            st.markdown("---")
            
            # Clear history button
            if st.button(self.get_text('clear_history')):
                st.session_state.chat_history = []
                st.session_state.query_submitted = False
                if 'selected_history_index' in st.session_state:
                    del st.session_state.selected_history_index
                st.rerun()
            
            # Chat history with clickable items
            if st.session_state.chat_history:
                st.markdown(f"### {self.get_text('recent_queries')}")
                
                # Add "Show Current" button if viewing history
                if st.session_state.get('selected_history_index') is not None:
                    current_label = "← Back to Current" if st.session_state.language == 'English' else "← 現在に戻る"
                    if st.button(current_label, key="back_to_current"):
                        if 'selected_history_index' in st.session_state:
                            del st.session_state.selected_history_index
                        st.rerun()
                    st.markdown("---")
                
                # Display chat history items (most recent first)
                recent_chats = list(reversed(st.session_state.chat_history[-10:]))  # Show last 10
                
                for idx, chat in enumerate(recent_chats):
                    # Create a unique key for each history item
                    history_key = f"history_{len(st.session_state.chat_history) - idx - 1}"
                    
                    # Truncate query for display
                    display_query = chat['query'][:40] + "..." if len(chat['query']) > 40 else chat['query']
                    
                    # Check if this item is selected
                    is_selected = st.session_state.get('selected_history_index') == (len(st.session_state.chat_history) - idx - 1)
                    
                    # Style the button differently if selected
                    button_style = "primary" if is_selected else "secondary"
                    
                    if st.button(
                        f"🕒 {chat['timestamp']} - {display_query}", 
                        key=history_key,
                        type=button_style,
                        use_container_width=True
                    ):
                        # Set selected history index
                        st.session_state.selected_history_index = len(st.session_state.chat_history) - idx - 1
                        st.rerun()
                    
                    # Show a small preview if not selected
                    #if not is_selected:
                        #with st.expander("Preview", expanded=False):
                         #  st.write(f"**Q:** {chat['query'][:100]}...")
                           # st.write(f"**Docs:** {chat['doc_count']} | **Images:** {chat['image_count']}")

    def render_main_interface(self):
        """Updated main interface to handle history viewing"""
        # Render sidebar
        self.render_sidebar()
        
        # Check if we're viewing a specific history item
        if st.session_state.get('selected_history_index') is not None:
            selected_index = st.session_state.selected_history_index
            
            if 0 <= selected_index < len(st.session_state.chat_history):
                selected_chat = st.session_state.chat_history[selected_index]
                
                # Header for history view
                history_header = "📜 Chat History" if st.session_state.language == 'English' else "📜 チャット履歴"
                st.markdown(f"## {history_header}")
                
                # Display the selected conversation
                self.display_chat_history_conversation(selected_chat)
                
                # Add option to ask a follow-up question
                st.markdown("---")
                followup_label = "Ask a follow-up question:" if st.session_state.language == 'English' else "フォローアップ質問をする:"
                st.markdown(f"### {followup_label}")
                
                with st.form(key="followup_form"):
                    followup_query = st.text_input(
                        "",
                        placeholder=self.get_text('input_placeholder'),
                        key="followup_query"
                    )
                    
                    ask_label = "🔍 Ask Follow-up" if st.session_state.language == 'English' else "🔍 フォローアップ質問"
                    followup_submit = st.form_submit_button(ask_label, type="primary")
                
                if followup_submit and followup_query.strip():
                    # Clear history selection and process new query
                    if 'selected_history_index' in st.session_state:
                        del st.session_state.selected_history_index
                    
                    # Process the follow-up query
                    response_container = st.container()
                    self.process_query(followup_query, response_container)
                
                return  # Exit early when viewing history
        
        # Show welcome screen only if no query has been submitted
        if not st.session_state.get('query_submitted', False):
            st.markdown(f"""
            <div style="text-align: center; padding: 2rem 0;">
                <h1 style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
                        font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem;">
                    {self.get_text('title')}
                </h1>
                <p style="color: #6c757d; font-size: 1.1rem; margin-bottom: 2rem;">
                    {self.get_text('subtitle')}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Center the start button
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("🔍 Start Searching", type="primary", use_container_width=True):
                    st.session_state.query_submitted = True
                    st.rerun()
            
            # Quick examples
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown(self.get_text('quick_examples'))
            
            examples = self.get_text('examples')
            
            # Check if any example button was clicked
            cols = st.columns(2)
            for idx, (category, example) in enumerate(examples):
                with cols[idx % 2]:
                    if st.button(f"{category}: {example}", key=f"example_{idx}"):
                        # Store the example query and navigate to search interface
                        st.session_state.selected_example = example
                        st.session_state.query_submitted = True
                        st.session_state.auto_submit_example = True
                        st.rerun()
        
        else:
            # Main search interface - shown after "Start Searching" is clicked
            
            # Always show the search form at the top for consistency
            with st.form(key="query_form"):
                query = st.text_input(
                    "",
                    placeholder=self.get_text('input_placeholder'),
                    key="main_query_form"
                )
                
                submit_clicked = st.form_submit_button(self.get_text('ask_button'), type="primary", use_container_width=True)
            
            # Check if we need to auto-submit an example
            if st.session_state.get('auto_submit_example') and st.session_state.get('selected_example'):
                # Process the example query immediately
                example_query = st.session_state.selected_example
                
                # Clear the example flags
                st.session_state.auto_submit_example = False
                if 'selected_example' in st.session_state:
                    del st.session_state.selected_example
                
                # Set flag to indicate we've processed a query
                st.session_state.first_query_processed = True
                
                # Process the example query
                response_container = st.container()
                self.process_query(example_query, response_container)
            
            # Handle manual form submission
            if submit_clicked:
                if query and query.strip():
                    # Create response container
                    response_container = st.container()
                    self.process_query(query, response_container)
                    st.session_state.first_query_processed = True
                else:
                    st.warning(self.get_text('enter_question'))


def main():
    """Main application entry point"""
    try:
        chatbot = ProfessionalChatbot()
        chatbot.render_main_interface()
    except Exception as e:
        error_text = "Failed to initialize the application. Please contact support." if st.session_state.get('language', 'English') == 'English' else "アプリケーションの初期化に失敗しました。サポートにお問い合わせください。"
        st.error(error_text)
        logger.error(f"Application initialization error: {e}")

if __name__ == "__main__":
    main()
