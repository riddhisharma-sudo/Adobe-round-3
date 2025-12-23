from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
import shutil
import time
from datetime import datetime
from typing import List
from dotenv import load_dotenv
from Adobe_Round_1A.documentExtraction import SmartPDFOutline
from Adobe_Round_1B.documentIntellligence import DocumentIntelligence
from src.multithreaded_processor import MultithreadedPDFProcessor
from src.vector_db import vector_db
from src.generate_insights import generate_insights
import uuid

# Load environment variables
load_dotenv()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for frontend
# static_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "dist")
# if os.path.exists(static_path):
#     app.mount("/", StaticFiles(directory=static_path, html=True), name="static")

# Request model for processing data
class ProcessRequest(BaseModel):
    persona: str
    jobToBeDone: str

# Request model for cross-PDF search
class CrossPDFSearchRequest(BaseModel):
    content: str  # Selected content to analyze

# Request model for PDF navigation with section search
class PDFNavigationRequest(BaseModel):
    document: str
    section_title: str
    original_text: str = ""  # Original text content for highlighting
    page_number: int
    switch_pdf: bool = True  # Whether to switch to this PDF first
    

static_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "dist")
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")

assets_path = os.path.join(static_path, "assets")

# serve index.html on root so frontend still loads
from fastapi.responses import FileResponse
if os.path.exists(static_path):
    # serve the index at /
    @app.get("/")
    def serve_index():
        return FileResponse(os.path.join(static_path, "index.html"))

    # serve the JS/CSS/images from dist/assets at /assets/*
    if os.path.exists(assets_path):
        app.mount("/assets", StaticFiles(directory=assets_path), name="assets")
    else:
        print("Warning: assets directory not found:", assets_path)

@app.get("/api/config")
def get_config():
    return {
        "adobeKey": os.getenv("ADOBE_EMBED_API_KEY"),
        "llm_provider": os.getenv("LLM_PROVIDER"),
    }


@app.post("/api/process")
async def process_data(request: ProcessRequest):
    """Process persona and job-to-be-done to return relevant PDF data"""
    try:
        # Read the JSON data
        file_path = os.path.join(os.getcwd(), "new.json")
        if not os.path.exists(file_path):
            return {"error": "Data file not found"}
        
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        return {
            "metadata": data.get("metadata", {}),
            "subsection_analysis": data.get("subsection_analysis", []),
            "request": {
                "persona": request.persona,
                "jobToBeDone": request.jobToBeDone
            }
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/get-json")
async def get_json():
    file_path = os.path.join(os.getcwd(), "data.json")
    if os.path.exists(file_path):
        # Read and return JSON directly instead of FileResponse for better CORS handling
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    return {"error": "File not found"}


@app.get("/api/get-pdfs")
async def get_pdfs():
    """Get list of available PDFs from the public assets directory"""
    try:
        # Path to the frontend public assets directory
        assets_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist", "assets")
        assets_path = os.path.normpath(assets_path)
        
        if not os.path.exists(assets_path):
            return {"error": "Assets directory not found", "pdfs": []}
        
        # Get all PDF files
        pdf_files = []
        for filename in os.listdir(assets_path):
            if filename.lower().endswith('.pdf'):
                pdf_files.append({
                    "name": filename,
                    "displayName": filename.replace('.pdf', '').replace('_', ' ').replace('-', ' '),
                    "path": f"assets/{filename}"
                })
        
        return {"pdfs": pdf_files}
    except Exception as e:
        return {"error": str(e), "pdfs": []}


@app.post("/api/upload-pdfs")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    """Upload PDFs to the assets folder"""
    try:
        # Path to assets folder
        assets_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist", "assets")
        assets_path = os.path.normpath(assets_path)
        
        # Create assets directory if it doesn't exist
        os.makedirs(assets_path, exist_ok=True)
        
        uploaded_files = []
        failed_files = []
        
        for file in files:
            try:
                # Check if it's a PDF
                if not file.filename.lower().endswith('.pdf'):
                    failed_files.append({
                        "filename": file.filename,
                        "error": "Not a PDF file"
                    })
                    continue
                
                # Save the file
                file_path = os.path.join(assets_path, file.filename)
                
                # Handle duplicates by adding a number suffix
                base_name = os.path.splitext(file.filename)[0]
                extension = os.path.splitext(file.filename)[1]
                counter = 1
                
                while os.path.exists(file_path):
                    new_filename = f"{base_name}_{counter}{extension}"
                    file_path = os.path.join(assets_path, new_filename)
                    counter += 1
                
                # Write the file
                with open(file_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)
                
                uploaded_files.append({
                    "original_name": file.filename,
                    "saved_name": os.path.basename(file_path),
                    "path": file_path
                })
                
            except Exception as e:
                failed_files.append({
                    "filename": file.filename,
                    "error": str(e)
                })
        
        return {
            "uploaded": uploaded_files,
            "failed": failed_files,
            "total_uploaded": len(uploaded_files),
            "total_failed": len(failed_files)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/extract-and-vectorize")
async def extract_and_vectorize():
    """Extract and chunk only new PDFs from assets and add to vector database incrementally"""
    try:
        # Paths
        assets_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist", "assets")
        assets_path = os.path.normpath(assets_path)
        
        if not os.path.exists(assets_path):
            return {"error": "Assets directory not found"}
        
        # Get all PDF files from assets
        pdf_files = []
        pdf_filenames = []
        
        for filename in os.listdir(assets_path):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(assets_path, filename)
                if os.path.isfile(pdf_path):
                    pdf_files.append(pdf_path)
                    pdf_filenames.append(filename)
        
        if not pdf_files:
            return {"error": "No PDF files found in assets"}
        
        # Check which documents are already processed
        processed_docs = vector_db.get_processed_documents()
        print(f"Already processed documents: {processed_docs}")
        
        # Filter to only new PDFs that need processing
        new_pdf_files = []
        new_pdf_filenames = []
        
        for pdf_path, pdf_filename in zip(pdf_files, pdf_filenames):
            if pdf_filename not in processed_docs:
                new_pdf_files.append(pdf_path)
                new_pdf_filenames.append(pdf_filename)
                print(f"Will process new PDF: {pdf_filename}")
            else:
                print(f"Skipping already processed PDF: {pdf_filename}")
        
        if not new_pdf_files:
            # No new PDFs to process
            stats = vector_db.get_database_stats()
            return {
                "success": True,
                "message": "No new PDFs to process - all are already vectorized",
                "processed_pdfs": [],
                "skipped_pdfs": pdf_filenames,
                "total_chunks": stats['total_chunks'],
                "vector_db_stats": stats
            }
        
        # Only process new PDFs (don't clear the database)
        print(f"Processing {len(new_pdf_files)} new PDFs...")
        
        # Initialize DocumentIntelligence for extraction
        di = DocumentIntelligence(content="Document extraction for vector database")
        
        # Extract and chunk only the new PDFs
        print("Starting PDF extraction and chunking...")
        extracted_data = di._extract_and_chunk_all(new_pdf_files)

        output_extracted_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "output_extracted.json")
        # Save extracted data to JSON file (append new data)
        existing_data = []
        if os.path.exists(output_extracted_path):
            try:
                with open(output_extracted_path, "r", encoding='utf-8') as f:
                    existing_data = json.load(f)
            except:
                existing_data = []
        
        # Combine existing and new data
        combined_data = existing_data + extracted_data
        with open(output_extracted_path, "w", encoding='utf-8') as f:
            json.dump(combined_data, f, indent=2, ensure_ascii=False)
        
        # Add only new chunks to vector database
        print("Adding new chunks to vector database...")
        chunk_ids = vector_db.add_document_chunks(extracted_data)
        
        # Get database statistics
        stats = vector_db.get_database_stats()
        
        print(f"Vector database updated with {len(chunk_ids)} new chunks from {len(new_pdf_files)} new PDFs")
        
        return {
            "success": True,
            "message": f"Successfully processed {len(new_pdf_files)} new PDFs",
            "processed_pdfs": new_pdf_filenames,
            "skipped_pdfs": [f for f in pdf_filenames if f not in new_pdf_filenames],
            "new_chunks": len(chunk_ids),
            "total_chunks": stats['total_chunks'],
            "vector_db_stats": stats
        }
        
    except Exception as e:
        print(f"Error in extract and vectorize: {str(e)}")
        return {"error": str(e)}

@app.post("/api/analyze-pdfs")
async def analyze_pdfs():
    """Analyze all PDFs from frontend assets, generate heading JSONs, extract chunks and create vector embeddings"""
    try:
        # Paths
        assets_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist", "assets")
        assets_path = os.path.normpath(assets_path)
        headings_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "heading_jsons")
        headings_dir = os.path.normpath(headings_dir)
        
        # Create heading_jsons directory if it doesn't exist
        os.makedirs(headings_dir, exist_ok=True)
        
        if not os.path.exists(assets_path):
            return {"error": "Assets directory not found", "processed": []}
        
        # Get all PDF files
        pdf_files = [f for f in os.listdir(assets_path) if f.lower().endswith(".pdf")]
        
        if not pdf_files:
            return {"message": "No PDF files found in assets", "processed": []}
        
        print(f"Found {len(pdf_files)} PDF files to analyze and vectorize...")
        
        # Check which documents are already processed
        processed_docs = vector_db.get_processed_documents()
        print(f"Already processed documents: {processed_docs}")
        
        # Filter to only new PDFs that need processing
        new_pdf_files = [f for f in pdf_files if f not in processed_docs]
        
        if not new_pdf_files:
            # No new PDFs to process
            stats = vector_db.get_database_stats()
            print("No new PDFs to vectorize - all are already processed")
            return {
                "message": "PDF analysis completed - no new files to process",
                "heading_analysis": {
                    "processed": [],
                    "failed": [],
                    "total_pdfs": len(pdf_files),
                    "successful": 0,
                    "failed_count": 0,
                    "skipped": len(pdf_files)
                },
                "vectorization": {
                    "success": True,
                    "message": "No new PDFs to process",
                    "processed_pdfs": [],
                    "skipped_pdfs": pdf_files,
                    "total_chunks": stats['total_chunks'],
                    "vector_db_stats": stats
                }
            }
        
        # Initialize multithreaded processor
        # Process 5 PDFs per thread for optimal performance
        processor = MultithreadedPDFProcessor(
            vector_db=vector_db,
            max_workers=None,  
            pdfs_per_thread=5
        )
        
        # Process all new PDFs using multithreading
        print(f"Starting multithreaded processing of {len(new_pdf_files)} new PDFs...")
        result = processor.process_all_pdfs(new_pdf_files, assets_path, headings_dir)
        
        if result["success"]:
            # Combine results
            return {
                "message": "PDF analysis and vectorization completed using multithreading",
                "heading_analysis": {
                    "processed": result["processed_files"],
                    "failed": result["failed_files"],
                    "total_pdfs": len(pdf_files),
                    "successful": len(result["processed_files"]),
                    "failed_count": len(result["failed_files"]),
                    "processing_time": result["processing_time"],
                    "performance": {
                        "multithreaded": True,
                        "pdfs_per_thread": 5,
                        "avg_time_per_pdf": result["processing_time"] / len(new_pdf_files) if new_pdf_files else 0
                    }
                },
                "vectorization": result["vectorization"]
            }
        else:
            return {
                "error": "Multithreaded processing failed",
                "details": result,
                "heading_analysis": {
                    "processed": result.get("processed_files", []),
                    "failed": result.get("failed_files", []),
                    "total_pdfs": len(pdf_files),
                    "successful": len(result.get("processed_files", [])),
                    "failed_count": len(result.get("failed_files", []))
                },
                "vectorization": {
                    "success": False,
                    "error": result.get("error", "Unknown error")
                }
            }
        
    except Exception as e:
        print(f"Overall error in analyze_pdfs: {str(e)}")
        return {"error": str(e), "processed": []}

@app.get("/api/get-pdf-headings/{pdf_name}")
async def get_pdf_headings(pdf_name: str):
    """Get the headings JSON for a specific PDF"""
    try:
        # Remove .pdf extension if present and add .json
        pdf_basename = pdf_name.replace('.pdf', '')
        json_filename = f"{pdf_basename}.json"
        
        headings_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "heading_jsons")
        headings_dir = os.path.normpath(headings_dir)
        json_path = os.path.join(headings_dir, json_filename)
        
        if not os.path.exists(json_path):
            return {"error": f"Headings not found for {pdf_name}. Run analysis first.", "headings": None}
        
        with open(json_path, 'r', encoding='utf-8') as file:
            headings_data = json.load(file)
        
        return {
            "pdf_name": pdf_name,
            "json_file": json_filename,
            "headings": headings_data
        }
        
    except Exception as e:
        return {"error": str(e), "headings": None}

@app.delete("/api/delete-pdf/{pdf_name}")
async def delete_pdf(pdf_name: str):
    """Delete a specific PDF and its corresponding JSON from assets and heading_jsons"""
    try:
        # Paths
        assets_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist", "assets")
        assets_path = os.path.normpath(assets_path)
        headings_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "heading_jsons")
        headings_dir = os.path.normpath(headings_dir)
        
        # PDF file path
        pdf_path = os.path.join(assets_path, pdf_name)
        # JSON file path
        json_filename = os.path.splitext(pdf_name)[0] + ".json"
        json_path = os.path.join(headings_dir, json_filename)
        
        deleted_files = []
        errors = []
        
        # Delete PDF file
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
            deleted_files.append(f"PDF: {pdf_name}")
            print(f"Deleted PDF: {pdf_path}")
        else:
            errors.append(f"PDF file not found: {pdf_name}")
        
        # Delete JSON file
        if os.path.exists(json_path):
            os.remove(json_path)
            deleted_files.append(f"JSON: {json_filename}")
            print(f"Deleted JSON: {json_path}")
        else:
            errors.append(f"JSON file not found: {json_filename}")
        
        return {
            "message": f"Deletion completed for {pdf_name}",
            "deleted_files": deleted_files,
            "errors": errors,
            "success": len(deleted_files) > 0
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/delete-all-pdfs")
async def delete_all_pdfs():
    """Delete all PDFs and their corresponding JSONs from assets and heading_jsons"""
    try:
        # Paths
        assets_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist", "assets")
        assets_path = os.path.normpath(assets_path)
        headings_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "heading_jsons")
        headings_dir = os.path.normpath(headings_dir)
        
        deleted_files = []
        errors = []
        
        # Delete all PDF files
        if os.path.exists(assets_path):
            pdf_files = [f for f in os.listdir(assets_path) if f.lower().endswith(".pdf")]
            for pdf_file in pdf_files:
                try:
                    pdf_path = os.path.join(assets_path, pdf_file)
                    os.remove(pdf_path)
                    deleted_files.append(f"PDF: {pdf_file}")
                    print(f"Deleted PDF: {pdf_path}")
                except Exception as e:
                    errors.append(f"Failed to delete PDF {pdf_file}: {str(e)}")
        
        # Delete all JSON files
        if os.path.exists(headings_dir):
            json_files = [f for f in os.listdir(headings_dir) if f.lower().endswith(".json")]
            for json_file in json_files:
                try:
                    json_path = os.path.join(headings_dir, json_file)
                    os.remove(json_path)
                    deleted_files.append(f"JSON: {json_file}")
                    print(f"Deleted JSON: {json_path}")
                except Exception as e:
                    errors.append(f"Failed to delete JSON {json_file}: {str(e)}")
        
        # Clear the vector database
        try:
            vector_db.clear_database()
            print("Vector database cleared successfully")
            deleted_files.append("Vector database cleared")
        except Exception as e:
            errors.append(f"Failed to clear vector database: {str(e)}")
        
        return {
            "message": f"Deleted {len(deleted_files)} files and cleared vector database",
            "deleted_files": deleted_files,
            "errors": errors,
            "success": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/cross-pdf-search")
async def cross_pdf_search(request: CrossPDFSearchRequest):
    """Search across all PDFs using enhanced context-aware search (semantic + keyword + structure)"""

    try:
        # Check if vector database has data
        db_stats = vector_db.get_database_stats()
        if db_stats['total_chunks'] == 0:
            return {"error": "Vector database is empty. Please run 'Extract and Vectorize' first."}
        
        print(f"Searching vector database with {db_stats['total_chunks']} chunks using enhanced search...")
        
        # Initialize enhanced search
        from src.enhanced_search import EnhancedSearch
        enhanced_searcher = EnhancedSearch(vector_db)
        
        # Perform intent-aware hybrid search
        print(f"Searching for: {request.content}")
        enhanced_chunks = enhanced_searcher.intent_aware_search(request.content, top_k=10)
        
        if not enhanced_chunks:
            return {"error": "No similar content found in the database"}
        
        print(f"Found {len(enhanced_chunks)} relevant chunks using enhanced context-aware search")
        
        # Create the output in the required format using enhanced results
        from datetime import datetime
        current_time = datetime.now().isoformat()
        
        # Create subsection analysis with enhanced context information
        subsection_analysis = []
        for i, chunk in enumerate(enhanced_chunks):
            # Create enhanced context description
            context_info = []
            if chunk.get('matching_keywords'):
                context_info.append(f"Keywords: {', '.join(chunk['matching_keywords'])}")
            if chunk.get('intent'):
                context_info.append(f"Intent: {chunk['intent']}")
            if chunk.get('merged_chunks', 0) > 1:
                context_info.append(f"Merged {chunk['merged_chunks']} chunks")
            
            context_description = f"Enhanced search (Hybrid: {chunk['hybrid_score']:.3f}, Intent: {chunk.get('intent_score', 0):.2f})"
            if context_info:
                context_description += f" - {'; '.join(context_info)}"
            
            subsection_analysis.append({
                "section_title": chunk.get('heading', f"Section {i+1}"),
                "document": chunk['doc'],
                "page_number": chunk['page'],
                "Original_text": chunk['text'],  
                "refined_text": chunk['text'], 
                "similarity_score": chunk['similarity_score'],
                "hybrid_score": chunk['hybrid_score'],
                "keyword_score": chunk.get('keyword_score', 0),
                "structure_score": chunk.get('structure_score', 0),
                "intent_score": chunk.get('intent_score', 0),
                "vector_rank": chunk['rank'],
                "matching_keywords": chunk.get('matching_keywords', []),
                "intent": chunk.get('intent', 'general'),
                "merged_chunks": chunk.get('merged_chunks', 1),
                "relevance_context": context_description
            })
        
        output_data = {
            "metadata": {
                "input_documents": db_stats['documents'],
                "content": request.content,
                "processing_timestamp": current_time,
                "search_method": "enhanced_context_aware_search",
                "search_components": ["semantic_similarity", "keyword_matching", "structure_analysis", "intent_detection"],
                "total_chunks_searched": db_stats['total_chunks'],
                "relevant_chunks_found": len(enhanced_chunks),
                "processing_note": "Results using enhanced context-aware search with hybrid scoring"
            },
            "subsection_analysis": subsection_analysis,
            "enhanced_search_stats": {
                "avg_hybrid_score": sum(c['hybrid_score'] for c in enhanced_chunks) / len(enhanced_chunks),
                "avg_keyword_score": sum(c.get('keyword_score', 0) for c in enhanced_chunks) / len(enhanced_chunks),
                "avg_structure_score": sum(c.get('structure_score', 0) for c in enhanced_chunks) / len(enhanced_chunks),
                "detected_intent": enhanced_chunks[0].get('intent', 'general') if enhanced_chunks else 'unknown',
                "unique_keywords": list(set([kw for c in enhanced_chunks for kw in c.get('matching_keywords', [])])),
                "total_merged_chunks": sum(c.get('merged_chunks', 1) for c in enhanced_chunks),
                "unique_documents": len(set(c['doc'] for c in enhanced_chunks)),
                "unique_sections": len(set(f"{c['doc']}::{c.get('heading', '')}" for c in enhanced_chunks))
            },
            "vector_search_results": [
                {
                    "text": chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text'],
                    "doc": chunk['doc'],
                    "page": chunk['page'],
                    "heading": chunk['heading'],
                    "similarity_score": chunk['similarity_score'],
                    "hybrid_score": chunk['hybrid_score'],
                    "rank": chunk['rank']
                } for chunk in enhanced_chunks
            ]
        }
        
        # Generate output filename based on content and timestamp
        safe_content = "".join(c for c in request.content[:50] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_content = safe_content.replace(' ', '_').lower() or "content_analysis"
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_filename = f"navigation_analysis_{safe_content}_{timestamp}.json"
        
        # Save to navigation_json directory
        output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "navigation_json")
        print(os.path.dirname(__file__))
        output_dir = os.path.normpath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        
        with open(output_path, "w", encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"Semantic search analysis complete. Results saved to {output_path}")
        
        # Return the analysis result
        return {
            "success": True,
            "output_file": output_filename,
            "data": output_data
        }
        
    except Exception as e:
        print(f"Error in cross-PDF search: {str(e)}")
        return {"error": str(e)}

@app.post("/api/navigate-to-section")
async def navigate_to_section(request: PDFNavigationRequest):
    """Navigate to a specific section in a PDF document"""
    try:
        # Paths
        assets_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist", "assets")
        assets_path = os.path.normpath(assets_path)
        headings_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "heading_jsons")
        headings_dir = os.path.normpath(headings_dir)
        
        if not os.path.exists(assets_path):
            return {"error": "Assets directory not found"}
            
        # Check if the PDF exists
        pdf_path = os.path.join(assets_path, request.document)
        if not os.path.exists(pdf_path):
            return {"error": f"PDF file '{request.document}' not found"}
        
        # Check if heading JSON exists
        json_filename = os.path.splitext(request.document)[0] + ".json"
        json_path = os.path.join(headings_dir, json_filename)
        
        section_info = None
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    heading_data = json.load(f)
                
                # Search for the section title in the outline
                for section in heading_data.get("outline", []):
                    if section.get("text", "").lower() == request.section_title.lower():
                        section_info = {
                            "found": True,
                            "exact_page": section.get("page"),
                            "level": section.get("level"),
                            "title": section.get("text")
                        }
                        break
                
                # If exact match not found, search for partial matches
                if not section_info:
                    for section in heading_data.get("outline", []):
                        if request.section_title.lower() in section.get("text", "").lower():
                            section_info = {
                                "found": True,
                                "exact_page": section.get("page"),
                                "level": section.get("level"),
                                "title": section.get("text"),
                                "match_type": "partial"
                            }
                            break
                            
            except Exception as e:
                print(f"Error reading heading JSON: {str(e)}")
        
        # Return navigation information
        result = {
            "success": True,
            "document": request.document,
            "section_title": request.section_title,
            "original_text": request.original_text,
            "requested_page": request.page_number,
            "pdf_exists": True,
            "pdf_path": f"assets/{request.document}", 
            "switch_pdf": request.switch_pdf,
            "highlight_text": request.section_title, 
            "highlight_content": request.original_text, 
            "section_info": section_info or {
                "found": False,
                "message": "Section title not found in document outline"
            }
        }
        
        # If we found the section in headings, use that page, otherwise use requested page
        if section_info and section_info.get("found"):
            result["recommended_page"] = section_info.get("exact_page", request.page_number)
        else:
            result["recommended_page"] = request.page_number
            
        return result
        
    except Exception as e:
        print(f"Error in PDF navigation: {str(e)}")
        return {"error": str(e)}

# Request model for saving selected content
class SaveSelectedContentRequest(BaseModel):
    filename: str
    content: dict

@app.post("/api/process-selected-content")
async def save_selected_content(request: SaveSelectedContentRequest):
    """Save selected content to get_content directory in outputs folder"""
    try:
        # Create get_content directory in outputs folder
        project_root = os.path.dirname(os.getcwd())  # Go up one level from backend
        get_content_dir = os.path.join(project_root, "outputs", "get_content")
        os.makedirs(get_content_dir, exist_ok=True)
        
        # Create full file path
        file_path = os.path.join(get_content_dir, request.filename)
        
        # Save the content as JSON
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(request.content, f, indent=2, ensure_ascii=False)
        

        return {
            "success": True,
            "message": "Content saved successfully",
            "filepath": file_path,
            "filename": request.filename
        }
        
    except Exception as e:
        print(f"Error saving selected content: {str(e)}")
        return {"error": str(e)}

@app.get("/api/vector-db-stats")
async def get_vector_db_stats():
    """Get statistics about the vector database"""
    try:
        stats = vector_db.get_database_stats()
        return {
            "success": True,
            "stats": stats
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/processed-documents")
async def get_processed_documents():
    """Get a list of documents that are already processed in the vector database"""
    try:
        processed_docs = vector_db.get_processed_documents()
        stats = vector_db.get_database_stats()
        return {
            "success": True,
            "processed_documents": processed_docs,
            "total_documents": len(processed_docs),
            "total_chunks": stats['total_chunks']
        }
    except Exception as e:
        return {"error": str(e)}

@app.delete("/api/remove-document/{document_name}")
async def remove_document(document_name: str):
    """Remove a specific document from the vector database"""
    try:
        removed_chunks = vector_db.remove_document(document_name)
        stats = vector_db.get_database_stats()
        return {
            "success": True,
            "message": f"Removed {removed_chunks} chunks for document: {document_name}",
            "removed_chunks": removed_chunks,
            "remaining_stats": stats
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/rebuild-vector-db")
async def rebuild_vector_db():
    """Force rebuild the entire vector database from all PDFs"""
    try:
        # Clear the database
        vector_db.clear_database()
        
        # Call the extract and vectorize endpoint
        result = await extract_and_vectorize()
        
        return {
            "success": True,
            "message": "Vector database rebuilt successfully",
            "rebuild_result": result
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/vector-search")
async def vector_search(request: CrossPDFSearchRequest):
    """Simple vector search without LLM analysis"""
    try:
        # Check if vector database has data
        db_stats = vector_db.get_database_stats()
        if db_stats['total_chunks'] == 0:
            return {"error": "Vector database is empty. Please run 'Extract and Vectorize' first."}
        
        # Search for similar chunks
        similar_chunks = vector_db.search_similar(request.content, top_k=10)
        
        return {
            "success": True,
            "query": request.content,
            "total_chunks_searched": db_stats['total_chunks'],
            "results_found": len(similar_chunks),
            "results": similar_chunks
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.delete("/api/clear-vector-db")
async def clear_vector_database():
    """Clear all data from the vector database"""
    try:
        vector_db.clear_database()
        return {
            "success": True,
            "message": "Vector database cleared successfully"
        }
    except Exception as e:
        return {"error": str(e)}
    


# Request model for insights generation from selected content
class GenerateInsightsFromContentRequest(BaseModel):
    selected_text: str

@app.post("/api/generate-insights-from-content")
async def generate_insights_from_content_endpoint(request: GenerateInsightsFromContentRequest):
    """Generate insights from selected text and similar content from vector database"""
    try:
        from src.generate_insights import generate_insights
        
        # Validate input
        if not request.selected_text or len(request.selected_text.strip()) < 1:
            return {"error": "No text selected. Please select text from the PDF to generate insights."}
        
        # Check if vector database has data
        db_stats = vector_db.get_database_stats()
        if db_stats['total_chunks'] == 0:
            return {"error": "Vector database is empty. Please run 'Extract and Vectorize' first to analyze documents."}
        
        print(f"Generating insights from selected text: {request.selected_text[:100]}...")
        print(f"Searching vector database with {db_stats['total_chunks']} chunks for similar content...")
        
        # Search for similar chunks using vector similarity
        similar_chunks = vector_db.search_similar(request.selected_text, top_k=5)
        
        # Combine selected text with similar content
        combined_content = f"Selected Content:\n{request.selected_text}\n\n"
        
        if similar_chunks:
            combined_content += "Related Content from Documents:\n"
            for i, chunk in enumerate(similar_chunks, 1):
                combined_content += f"\n{i}. From {chunk['doc']} (Page {chunk['page']}):\n"
                combined_content += f"   Section: {chunk.get('heading', 'N/A')}\n"
                combined_content += f"   Content: {chunk['text']}\n"
                combined_content += f"   Similarity Score: {chunk['similarity_score']:.3f}\n"
        else:
            combined_content += "No similar content found in the document database.\n"
        
        print(f"Combined content length: {len(combined_content)} characters")
        print(f"Found {len(similar_chunks)} similar chunks from vector database")
        
        # Create insights output directory
        insights_output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "insights_json")
        insights_output_dir = os.path.normpath(insights_output_dir)
        os.makedirs(insights_output_dir, exist_ok=True)
        
        # Generate timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        insights_json_path = os.path.join(insights_output_dir, f"insights_analysis_{timestamp}.json")
        
        # Create the JSON structure expected by generate_insights function
        insights_input_data = {
            "selected_content": request.selected_text,
            "similar_chunks": similar_chunks,
            "combined_analysis": combined_content,
            "metadata": {
                "total_chunks_searched": db_stats['total_chunks'],
                "similar_chunks_found": len(similar_chunks),
                "content_length": len(combined_content),
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # Save temporary input file for insights generation
        temp_dir = os.path.join(os.path.dirname(__file__), "temp")
        os.makedirs(temp_dir, exist_ok=True)
        temp_input_path = os.path.join(temp_dir, f"temp_insights_input_{timestamp}.json")
        
        with open(temp_input_path, 'w', encoding='utf-8') as f:
            json.dump(insights_input_data, f, indent=2, ensure_ascii=False)
        
        print("Generating insights from selected text and similar content...")
        insights_data = generate_insights(temp_input_path)
        
        if not insights_data:
            # Clean up temporary file
            try:
                os.remove(temp_input_path)
            except:
                pass
            return {"error": "Failed to generate insights from the selected content."}
        
        # Save the insights data to the output file
        with open(insights_json_path, 'w', encoding='utf-8') as f:
            json.dump(insights_data, f, indent=2, ensure_ascii=False)
        
        print(f"Insights generation completed successfully! Results saved to {insights_json_path}")
        
        # Clean up temporary file
        try:
            os.remove(temp_input_path)
        except:
            pass
        
        # Generate response with insights data
        timestamp_str = timestamp  # Use the same timestamp
        
        response_data = {
            "success": True,
            "message": "Insights generated successfully from selected content",
            "input_text_length": len(request.selected_text),
            "similar_chunks_found": len(similar_chunks),
            "combined_content_length": len(combined_content),
            "insights_json": f"insights_analysis_{timestamp}.json",
            "insights_data": insights_data,  # Include the actual insights data
            "timestamp": timestamp_str,
            "vector_search_info": {
                "total_chunks_searched": db_stats['total_chunks'],
                "similar_chunks_used": len(similar_chunks),
                "similarity_scores": [chunk['similarity_score'] for chunk in similar_chunks] if similar_chunks else []
            }
        }
        
        return response_data
        
    except Exception as e:
        print(f"Error generating insights from content: {str(e)}")
        return {"error": str(e)}


def cleanup_previous_podcast_files():
    """Clean up previous podcast generation files"""
    podcast_voice_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "podcast_voice")
    podcast_output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "podcast_output")
    os.makedirs(podcast_voice_dir, exist_ok=True)
    os.makedirs(podcast_output_dir, exist_ok=True)
    
    cleanup_files = []
    
    # Clean up audio files and related files from podcast_voice directory
    if os.path.exists(podcast_voice_dir):
        for filename in os.listdir(podcast_voice_dir):
            file_path = os.path.join(podcast_voice_dir, filename)
            if os.path.isfile(file_path):
                # Remove all audio files and related files from previous generation
                if (filename.endswith('.mp3') or 
                    filename == 'generation_report.json' or 
                    filename == 'podcast_playlist.txt'):
                    cleanup_files.append(file_path)
    
    # Clean up podcast.json from podcast_output directory
    if os.path.exists(podcast_output_dir):
        for filename in os.listdir(podcast_output_dir):
            file_path = os.path.join(podcast_output_dir, filename)
            if os.path.isfile(file_path):
                # Remove podcast.json and any other JSON files from previous generation
                if filename == 'podcast.json' or filename.endswith('.json'):
                    cleanup_files.append(file_path)
    
    # Delete the files
    deleted_count = 0
    for file_path in cleanup_files:
        try:
            os.remove(file_path)
            print(f"Deleted previous file: {os.path.basename(file_path)}")
            deleted_count += 1
        except Exception as e:
            print(f"Warning: Could not delete {os.path.basename(file_path)}: {str(e)}")
    
    if deleted_count > 0:
        print(f"Cleaned up {deleted_count} files from previous podcast generation")
        # Small delay to ensure files are completely deleted
        time.sleep(0.1)
    
    return deleted_count


# Request model for podcast generation  
class GeneratePodcastRequest(BaseModel):
    selected_text: str

@app.post("/api/generate-podcast")
async def generate_podcast_endpoint(request: GeneratePodcastRequest):
    """Generate podcast from selected text and similar content from vector database"""
    try:
        from src.generate_podcast import generate_podcast
        # Try the multithreaded version first, fallback to sequential if needed


        try:
            from src.multithreaded_podcast_generator import generate_simple_podcast
            use_simple_audio = True
        except ImportError:
            from src.sequential_podcast_generator import generate_json_podcast
            use_simple_audio = False
        
        # Validate input
        if not request.selected_text or len(request.selected_text.strip()) < 1:
            return {"error": "No text selected. Please select text from the PDF to generate a podcast."}
        
        # Check if vector database has data
        db_stats = vector_db.get_database_stats()
        if db_stats['total_chunks'] == 0:
            return {"error": "Vector database is empty. Please run 'Extract and Vectorize' first to analyze documents."}
        
        print(f"Generating podcast from selected text: {request.selected_text[:100]}...")
        print(f"Searching vector database with {db_stats['total_chunks']} chunks for similar content...")
        
        # Search for similar chunks using vector similarity
        similar_chunks = vector_db.search_similar(request.selected_text, top_k=5)
        
        # Combine selected text with similar content
        combined_content = f"Selected Content:\n{request.selected_text}\n\n"
        
        if similar_chunks:
            combined_content += "Related Content from Documents:\n"
            for i, chunk in enumerate(similar_chunks, 1):
                combined_content += f"\n{i}. From {chunk['doc']} (Page {chunk['page']}):\n"
                combined_content += f"   Section: {chunk.get('heading', 'N/A')}\n"
                combined_content += f"   Content: {chunk['text']}\n"
                combined_content += f"   Similarity Score: {chunk['similarity_score']:.3f}\n"
        else:
            combined_content += "No similar content found in the document database.\n"
        
        print(f"Combined content length: {len(combined_content)} characters")
        print(f"Found {len(similar_chunks)} similar chunks from vector database")
        
        # Clean up previous podcast files
        cleanup_previous_podcast_files()
        
        # Set up directories
        podcast_voice_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "podcast_voice")
        podcast_output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "podcast_output")
        os.makedirs(podcast_output_dir, exist_ok=True)
        
        # Generate timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        podcast_json_path = os.path.join(podcast_output_dir, "podcast.json")

        print("Generating podcast script from selected text and similar content...")
        podcast_data = generate_podcast(combined_content, podcast_json_path)
        
        # Save the podcast data to the file
        with open(podcast_json_path, 'w', encoding='utf-8') as f:
            json.dump(podcast_data, f, indent=2, ensure_ascii=False)
        
        if not os.path.exists(podcast_json_path):
            return {"error": "Failed to generate podcast JSON script."}
        
        print("Podcast JSON generated successfully!")
        
        # Try to generate audio file from podcast JSON, but handle errors gracefully
        audio_generated = False
        final_audio_path = os.path.join(podcast_voice_dir, "final_episode.mp3")

        try:
            print("Converting podcast script to audio using multithreaded generation...")
            if use_simple_audio:
                # Use multithreaded generation with 15 concurrent workers for faster processing
                audio_result = generate_simple_podcast(podcast_json_path, max_workers=10)
            else:
                audio_result = generate_json_podcast(podcast_json_path)
            
            if os.path.exists(final_audio_path):
                audio_generated = True
                print("Audio generation completed successfully!")
            else:
                print("Audio file was not created.")
                
        except Exception as audio_error:
            print(f"Audio generation failed (but JSON was created): {str(audio_error)}")
            # Audio generation failed, but we still have the JSON data
            audio_generated = False
        
        # Generate response with podcast data
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        
        response_data = {
            "success": True,
            "message": "Podcast script generated successfully",
            "input_text_length": len(request.selected_text),
            "similar_chunks_found": len(similar_chunks),
            "combined_content_length": len(combined_content),
            "podcast_json": "podcast.json",
            "podcast_data": podcast_data,  # Include the actual podcast script data
            "timestamp": timestamp_str,
            "audio_generated": audio_generated,
            "vector_search_info": {
                "total_chunks_searched": db_stats['total_chunks'],
                "similar_chunks_used": len(similar_chunks),
                "similarity_scores": [chunk['similarity_score'] for chunk in similar_chunks] if similar_chunks else []
            }
        }


        if audio_generated:
            # Add timestamp to audio URL for cache-busting
            audio_timestamp = int(time.time())
            response_data["audio_file"] = f"final_episode.mp3"
            response_data["audio_url"] = f"/api/podcast-audio?t={audio_timestamp}"
            response_data["audio_timestamp"] = audio_timestamp
            response_data["message"] = "Podcast generated successfully with audio"
        else:
            response_data["message"] = "Podcast script generated (audio generation skipped due to dependency issues)"
            response_data["note"] = "Audio generation requires additional dependencies. The podcast script is available as JSON."
        
        return response_data
        
    except Exception as e:
        print(f"Error generating podcast: {str(e)}")
        return {"error": str(e)}
        
      
@app.get("/api/podcast-audio")
async def get_podcast_audio(t: int = None):
    """Serve the generated podcast audio file with cache-busting"""
    try:
        podcast_output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "podcast_voice")
        audio_file_path = os.path.join(podcast_output_dir, "final_episode.mp3")
        
        print(f"Looking for podcast audio at: {audio_file_path} (cache-bust: {t})")
        
        if not os.path.exists(audio_file_path):
            raise HTTPException(status_code=404, detail=f"Podcast audio file not found at {audio_file_path}")
        
        # Get file modification time for ETag and Last-Modified headers
        file_stat = os.stat(audio_file_path)
        file_mtime = file_stat.st_mtime
        etag = f'"{file_mtime}"'
        last_modified = datetime.fromtimestamp(file_mtime).strftime('%a, %d %b %Y %H:%M:%S GMT')
        
        return FileResponse(
            audio_file_path,
            media_type="audio/mpeg",
            filename="final_episode.mp3",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
                "ETag": etag,
                "Last-Modified": last_modified
            }
        )
        
    except Exception as e:
        print(f"Error serving podcast audio: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/podcast-cleanup")
async def cleanup_podcast_files():
    """Manual endpoint to clean up previous podcast files"""
    try:
        deleted_count = cleanup_previous_podcast_files()
        
        return {
            "success": True,
            "message": f"Cleanup completed successfully",
            "files_deleted": deleted_count
        }
        
    except Exception as e:
        print(f"Error during manual cleanup: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))



