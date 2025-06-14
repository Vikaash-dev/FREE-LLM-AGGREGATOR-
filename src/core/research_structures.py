from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import uuid

@dataclass
class ResearchQuery:
    '''Represents the input query for the research assistant.'''
    query_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    original_task_description: str
    keywords: List[str] # Keywords extracted for searching
    project_context_summary: Optional[str] = None # A brief summary of the project context
    max_results_per_keyword: int = 3 # Default number of results to aim for per keyword

@dataclass
class SearchResult:
    '''Represents a raw search result, typically from a (mocked) search engine.'''
    result_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_identifier: str # e.g., URL for web, or an ID for mock content
    matched_keyword: Optional[str] = None
    title: Optional[str] = None # If available from search
    snippet: Optional[str] = None # If available from search

@dataclass
class ProcessedResult:
    '''Represents a search result after content has been fetched and relevance scored.'''
    processed_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    original_search_result: SearchResult
    fetched_content: str # The actual content fetched from the source
    relevance_score: float = 0.0 # Score from 0.0 to 1.0
    relevance_justification: Optional[str] = None # Explanation for the score
    error_fetching: Optional[str] = None # If content fetching failed

@dataclass
class KnowledgeChunk:
    '''Represents a piece of synthesized information derived from research.'''
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str # The synthesized piece of information (e.g., a key point, a summary paragraph)
    source_result_ids: List[str] = field(default_factory=list) # IDs of ProcessedResults it was derived from
    confidence_score: Optional[float] = None # Confidence in the accuracy/relevance of this chunk
