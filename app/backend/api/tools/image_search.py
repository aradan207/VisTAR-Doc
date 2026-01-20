"""
Image search tool - Semantic Search Version.

Uses VLM descriptions and embeddings for intelligent semantic image search.
Combines machine name filtering with embedding-based similarity for best results.

Features:
1. Semantic search using VLM description embeddings
2. Machine/PDF name filtering for relevance
3. Content type detection and filtering
4. Page proximity scoring
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
from pydantic import BaseModel, Field

from app.backend.core.agent.tool import tool


# Paths - resolve relative to this file's location
_TOOLS_DIR = Path(__file__).parent
_API_DIR = _TOOLS_DIR.parent
_BACKEND_DIR = _API_DIR.parent
_APP_DIR = _BACKEND_DIR.parent
_AGENTIC_RAG_DIR = _APP_DIR.parent
_REPOSITORIES_DIR = _AGENTIC_RAG_DIR.parent

YOLOGEN_DIR = _REPOSITORIES_DIR / "vlm-yolo-detector"
IMAGE_INDEX_PATH = YOLOGEN_DIR / "data" / "processed" / "image_index.json"
EMBEDDING_NPY_PATH = YOLOGEN_DIR / "data" / "processed" / "image_embeddings.npy"
EMBEDDING_MAPPING_PATH = YOLOGEN_DIR / "data" / "processed" / "embedding_mapping.json"
IMAGES_BASE_DIR = YOLOGEN_DIR / "data" / "processed" / "images"

# Cached data
_image_index: Dict[str, Any] = {}
_embeddings: Optional[np.ndarray] = None
_filenames: List[str] = []
_embedding_model = None
_loaded = False


def _load_all_data(force_reload: bool = False):
    """Load image index, embeddings, and model."""
    global _image_index, _embeddings, _filenames, _embedding_model, _loaded
    
    if _loaded and not force_reload:
        return
    
    # Reset if force reloading
    if force_reload:
        _image_index = {}
        _embeddings = None
        _filenames = []
        _embedding_model = None
    
    # Load image index
    if IMAGE_INDEX_PATH.exists():
        try:
            with open(IMAGE_INDEX_PATH, 'r', encoding='utf-8') as f:
                _image_index = json.load(f)
            print(f"[ImageSearch] Loaded {len(_image_index)} images from index")
        except Exception as e:
            print(f"[ImageSearch] Failed to load index: {e}")
    else:
        print(f"[ImageSearch] Index not found: {IMAGE_INDEX_PATH}")
    
    # Load embeddings
    if EMBEDDING_NPY_PATH.exists() and EMBEDDING_MAPPING_PATH.exists():
        try:
            _embeddings = np.load(EMBEDDING_NPY_PATH)
            with open(EMBEDDING_MAPPING_PATH, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
            _filenames = mapping.get("filenames", [])
            print(f"[ImageSearch] Loaded {len(_filenames)} embeddings ({_embeddings.shape})")
            
            # Normalize embeddings for cosine similarity
            norms = np.linalg.norm(_embeddings, axis=1, keepdims=True)
            _embeddings = _embeddings / (norms + 1e-10)
            print(f"[ImageSearch] Embeddings normalized")
        except Exception as e:
            print(f"[ImageSearch] Failed to load embeddings: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"[ImageSearch] Embeddings not found - semantic search disabled")
        print(f"[ImageSearch]   NPY path: {EMBEDDING_NPY_PATH} exists={EMBEDDING_NPY_PATH.exists()}")
        print(f"[ImageSearch]   Mapping path: {EMBEDDING_MAPPING_PATH} exists={EMBEDDING_MAPPING_PATH.exists()}")
    
    # Load embedding model (lazy)
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            print("[ImageSearch] Embedding model loaded")
        except ImportError:
            print("[ImageSearch] sentence-transformers not installed - semantic search disabled")
        except Exception as e:
            print(f"[ImageSearch] Failed to load model: {e}")
    
    # Only mark as loaded if we have ALL essential components for semantic search
    if _image_index and _embeddings is not None and _embedding_model is not None:
        _loaded = True
        print(f"[ImageSearch] Successfully initialized with semantic search")
        print(f"[ImageSearch]   Index: {len(_image_index)} images")
        print(f"[ImageSearch]   Embeddings: {_embeddings.shape}")
        print(f"[ImageSearch]   Filenames: {len(_filenames)}")
    else:
        # DON'T mark loaded on partial failure - allow retry on next request
        _loaded = False
        print(f"[ImageSearch] WARNING: Partial initialization - will retry on next request")
        print(f"[ImageSearch]   Index loaded: {bool(_image_index)}")
        print(f"[ImageSearch]   Embeddings loaded: {_embeddings is not None}")
        print(f"[ImageSearch]   Model loaded: {_embedding_model is not None}")


def _extract_machine_names(query: str) -> List[str]:
    """Extract potential machine/manual names from query."""
    machines = []
    query_upper = query.upper()
    
    # Pattern: Word-Word or Word_Word (like APSX-PIM, RSA-G2)
    pattern1 = re.findall(r'\b([A-Z][A-Z0-9]*[-_][A-Z0-9]+(?:[-_][A-Z0-9]+)*)\b', query_upper)
    machines.extend(pattern1)
    
    # Pattern: Word followed by numbers (like BOY35, BOY 35)
    pattern2 = re.findall(r'\b([A-Z]+)\s*(\d+)\b', query_upper)
    for word, num in pattern2:
        machines.append(f"{word}-{num}")
        machines.append(f"{word}{num}")
        machines.append(word)
    
    # Pattern: Standalone capitalized words 3+ chars (potential brand names)
    pattern3 = re.findall(r'\b([A-Z]{3,})\b', query_upper)
    machines.extend(pattern3)
    
    return list(set(m.strip() for m in machines if len(m) >= 2))


def _normalize_pdf_name(pdf_name: str) -> str:
    """Normalize PDF name for matching."""
    name = pdf_name.upper()
    name = re.sub(r'^MANUAL_', '', name)
    name = re.sub(r'_UNLOCKED$', '', name)
    name = re.sub(r'\.PDF$', '', name)
    return name


def _calculate_pdf_match_score(query_machines: List[str], pdf_name: str) -> float:
    """Calculate how well a PDF name matches extracted machine names (0-10)."""
    if not query_machines:
        return 0.0
    
    normalized_pdf = _normalize_pdf_name(pdf_name)
    score = 0.0
    
    for machine in query_machines:
        machine_upper = machine.upper()
        
        if machine_upper == normalized_pdf:
            score += 10.0
            continue
        
        if machine_upper in normalized_pdf:
            score += 5.0 + (len(machine_upper) / len(normalized_pdf)) * 3.0
            continue
        
        machine_parts = re.split(r'[-_\s]', machine_upper)
        pdf_parts = re.split(r'[-_\s]', normalized_pdf)
        
        for mp in machine_parts:
            if len(mp) >= 2:
                for pp in pdf_parts:
                    if mp in pp or pp in mp:
                        score += 2.0
                        break
    
    return min(score, 10.0)


def _detect_required_content_type(query: str) -> Optional[str]:
    """Detect if the query requires a specific content type."""
    if not query:
        return None
    
    query_lower = query.lower()
    
    type_patterns = {
        'schematic': ['wiring diagram', 'wiring', 'circuit diagram', 'circuit', 
                      'electrical diagram', 'electrical schematic', 'hydraulic diagram',
                      'hydraulic schematic', 'schematic'],
        'diagram': ['diagram', 'layout', 'flowchart', 'block diagram', 'system diagram',
                    'overview diagram', 'connector diagram'],
        'table': ['table', 'specification table', 'spec table', 'data table'],
        'procedure': ['procedure', 'step-by-step', 'instructions', 'how to'],
        'warning': ['warning', 'safety', 'caution', 'danger'],
        'photo': ['photo', 'photograph', 'picture of', 'image of', 'what does it look like'],
        'component': ['component', 'part diagram', 'assembly', 'exploded view', 'connector'],
    }
    
    for label, patterns in type_patterns.items():
        for pattern in patterns:
            if pattern in query_lower:
                return label
    
    return None


def _semantic_search(query: str, top_k: int = 50) -> List[tuple]:
    """
    Perform semantic search using embeddings.
    Returns list of (filename, similarity_score) tuples.
    """
    global _embeddings, _filenames, _embedding_model
    
    if _embedding_model is None or _embeddings is None or len(_filenames) == 0:
        print(f"[ImageSearch] _semantic_search: SKIPPING - embeddings not ready")
        print(f"[ImageSearch]   model={_embedding_model is not None}, embeddings={_embeddings is not None}, filenames={len(_filenames)}")
        return []
    
    try:
        # Encode query
        query_embedding = _embedding_model.encode([query], convert_to_numpy=True)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Compute similarities
        similarities = np.dot(_embeddings, query_embedding.T).flatten()
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [(
            _filenames[idx],
            float(similarities[idx])
        ) for idx in top_indices]
    
    except Exception as e:
        print(f"[ImageSearch] Semantic search error: {e}")
        return []


def _search_images(
    query: str = None,
    class_filter: str = None,
    pdf_filter: str = None,
    page_hint: int = None,
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    Search images using hybrid semantic + rule-based approach.
    
    1. Semantic search for initial candidates
    2. Apply PDF/machine filtering
    3. Apply content type filtering
    4. Apply page proximity scoring
    """
    global _embeddings, _embedding_model, _filenames
    
    _load_all_data()
    
    # Validate embedding state - force reload if missing
    if _embeddings is None or _embedding_model is None or len(_filenames) == 0:
        print(f"[ImageSearch] Embeddings missing after load, forcing reload...")
        print(f"[ImageSearch]   _embeddings: {_embeddings is not None}")
        print(f"[ImageSearch]   _embedding_model: {_embedding_model is not None}")
        print(f"[ImageSearch]   _filenames: {len(_filenames)}")
        _load_all_data(force_reload=True)
        
        # Check again after force reload
        if _embeddings is None or _embedding_model is None:
            print(f"[ImageSearch] ERROR: Embeddings still not loaded after force reload!")
    
    if not _image_index:
        return []
    
    query_machines = _extract_machine_names(query) if query else []
    if pdf_filter:
        query_machines.append(pdf_filter.upper())
        query_machines = list(set(query_machines))
    
    required_type = _detect_required_content_type(query)
    
    # Get semantic search candidates
    semantic_results = _semantic_search(query, top_k=100) if query else []
    semantic_scores = {filename: score for filename, score in semantic_results}
    
    # Compatible content types
    compatible_types = {
        'schematic': {'schematic', 'diagram'},
        'diagram': {'diagram', 'schematic'},
        'table': {'table', 'specification'},
        'specification': {'specification', 'table'},
        'procedure': {'procedure'},
        'warning': {'warning'},
        'photo': {'photo', 'component'},
        'component': {'component', 'photo', 'diagram'},
    }
    
    results = []
    
    for image_name, metadata in _image_index.items():
        score = 0.0
        
        pdf_name = metadata.get('pdf_name', '')
        page_num = metadata.get('page', 0)
        vlm_desc = metadata.get('vlm_description', '')
        keywords = metadata.get('keywords', [])
        extraction_type = metadata.get('extraction_type', '')
        
        # Get content type from keywords or VLM description
        content_type = _infer_content_type(vlm_desc, keywords)
        
        # === CONTENT TYPE FILTERING ===
        if required_type:
            allowed = compatible_types.get(required_type, {required_type})
            if content_type and content_type not in allowed:
                continue
            if content_type in allowed:
                score += 8.0
        
        # === SEMANTIC SIMILARITY SCORE (0-10) ===
        if image_name in semantic_scores:
            sem_score = semantic_scores[image_name]
            score += sem_score * 10.0  # Scale to 0-10
        
        # === PDF SOURCE MATCH ===
        pdf_match_score = _calculate_pdf_match_score(query_machines, pdf_name)
        score += pdf_match_score * 3.0  # Weight PDF match heavily
        
        # Penalize if machine names in query but no PDF match
        if query_machines and pdf_match_score == 0:
            score -= 8.0  # Strong penalty for wrong machine
        
        # === EXPLICIT PDF FILTER (Strict) ===
        if pdf_filter:
            if pdf_filter.lower() in pdf_name.lower():
                score += 10.0
            else:
                continue  # Skip if explicit filter doesn't match
        
        # === CLASS FILTER ===
        if class_filter:
            if content_type == class_filter.lower():
                score += 3.0
        
        # === PAGE PROXIMITY ===
        if page_hint and page_num:
            page_distance = abs(page_num - page_hint)
            if page_distance == 0:
                score += 8.0
            elif page_distance <= 2:
                score += 5.0
            elif page_distance <= 5:
                score += 2.0
            elif page_distance <= 10:
                score += 1.0
        
        # Skip low relevance
        if score < 2.0:
            continue
        
        split = metadata.get('split', 'train')
        results.append({
            "image_name": image_name,
            "pdf_source": pdf_name,
            "page_number": page_num,
            "content_type": content_type,
            "vlm_description": vlm_desc[:200] if vlm_desc else "",
            "split": split,
            "score": round(score, 2),
            "semantic_score": round(semantic_scores.get(image_name, 0) * 100, 1)
        })
    
    # Sort by score, then page number
    results.sort(key=lambda x: (-x['score'], x['page_number']))
    return results[:top_k]


def _infer_content_type(vlm_description: str, keywords: List[str]) -> str:
    """Infer content type from VLM description and keywords."""
    if not vlm_description:
        return "unknown"
    
    desc_lower = vlm_description.lower()
    
    # Check for specific content types in description
    type_indicators = {
        'schematic': ['schematic', 'wiring', 'circuit', 'electrical diagram'],
        'diagram': ['diagram', 'flowchart', 'block diagram', 'layout'],
        'table': ['table', 'chart showing', 'data table'],
        'photo': ['photograph', 'photo of', 'image shows', 'close-up'],
        'warning': ['warning', 'caution', 'danger', 'safety notice'],
        'procedure': ['step', 'instruction', 'procedure', 'how to'],
        'component': ['component', 'part', 'assembly', 'exploded'],
        'specification': ['specification', 'spec', 'parameter'],
    }
    
    for content_type, indicators in type_indicators.items():
        for indicator in indicators:
            if indicator in desc_lower:
                return content_type
    
    # Fallback to keywords
    if keywords:
        for kw in keywords:
            if kw in type_indicators:
                return kw
    
    return "image"


class ImageSearchArgs(BaseModel):
    """Arguments for image_search tool."""
    
    query: str = Field(
        ...,
        description="Search query including the MACHINE NAME and what you're looking for. "
                    "Example: 'APSX-PIM wiring diagram', 'BOY 35 hydraulic schematic'",
    )
    class_filter: Optional[str] = Field(
        default=None,
        description="Filter by image type: diagram, component, warning, procedure, specification, table, schematic, photo",
    )
    pdf_filter: Optional[str] = Field(
        default=None,
        description="Filter by manual/PDF name (e.g., 'APSX-PIM', 'BOY-35'). "
                    "CRITICAL: Use this to ensure images come from the correct manual.",
    )
    page_hint: Optional[int] = Field(
        default=None,
        description="Page number from manual_search results. Images near this page are prioritized.",
    )
    top_k: int = Field(
        default=3,
        description="Number of images to return",
    )


@tool(
    "image_search",
    ImageSearchArgs,
    "Search for technical images from equipment manuals using semantic search. "
    "Uses VLM descriptions and embeddings for intelligent matching. "
    "CRITICAL: Always include the machine name in query AND use pdf_filter to get images from the correct manual. "
    "Use page_hint if manual_search found content on a specific page."
)
def image_search(args: ImageSearchArgs) -> dict:
    """Search for images from equipment manuals using semantic search."""
    try:
        required_type = _detect_required_content_type(args.query)
        
        results = _search_images(
            query=args.query,
            class_filter=args.class_filter,
            pdf_filter=args.pdf_filter,
            page_hint=args.page_hint,
            top_k=args.top_k
        )
        
        if not results:
            message = f"No images found matching '{args.query}'"
            if args.pdf_filter:
                message += f" from {args.pdf_filter}"
            if required_type:
                message += f". No {required_type} images are indexed for this manual"
            if args.page_hint:
                message += f". Page {args.page_hint} may contain non-extractable graphics. Refer user to view the actual PDF page."
            
            return {
                "count": 0,
                "results": [],
                "message": message,
                "suggestion": f"Recommend user view page {args.page_hint} in the PDF directly." if args.page_hint else None
            }
        
        base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
        
        formatted_results = []
        for r in results:
            if r["score"] <= 0:
                continue
            
            formatted_results.append({
                "image_name": r["image_name"],
                "pdf_source": r["pdf_source"],
                "page_number": r["page_number"],
                "content_type": r["content_type"],
                "description": r["vlm_description"],
                "relevance_score": r["score"],
                "semantic_match": f"{r['semantic_score']}%",
                "url": f"{base_url}/api/media/yologen/{r['split']}/{r['image_name']}"
            })
        
        if not formatted_results:
            return {
                "count": 0,
                "results": [],
                "message": f"No relevant images found for '{args.query}'. Images don't match this machine/topic."
            }
        
        # Check if page_hint was specified but no images are near that page
        if args.page_hint and required_type:
            result_pages = [r["page_number"] for r in formatted_results]
            closest_page = min(result_pages, key=lambda p: abs(p - args.page_hint))
            distance_to_closest = abs(closest_page - args.page_hint)
            
            if distance_to_closest > 5:
                return {
                    "count": len(formatted_results),
                    "results": formatted_results,
                    "warning": (
                        f"No {required_type} images found near page {args.page_hint}. "
                        f"Returned images are from other pages. The specific {required_type} on page {args.page_hint} "
                        f"may be a vector graphic not extracted. Please refer user to view that PDF page directly."
                    ),
                    "page_reference": args.page_hint
                }
        
        return {
            "count": len(formatted_results),
            "results": formatted_results
        }
        
    except Exception as e:
        return {"error": f"Image search failed: {str(e)}", "results": []}
