from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import uuid

@dataclass
class ResearchQuery: # No changes to this usually, but review if Tavily needs specific query fields
    '''Represents the input query for the research assistant.'''
    query_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    original_task_description: str # This will be the main query for Tavily
    keywords: List[str] = field(default_factory=list) # Can be used to refine query for Tavily or as fallback
    project_context_summary: Optional[str] = None
    # Tavily specific parameters can be added here or passed directly to search method
    # search_depth: str = "basic"
    # max_results: int = 5
    # include_answer: bool = False
    # include_raw_content: bool = False

@dataclass
class WebSearchResult: # Renamed from SearchResult
    '''
    Represents a single search result item, adapted for Tavily's output or general web results.
    '''
    result_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    url: str # URL of the search result, core from Tavily
    title: Optional[str] = None # Title of the page, usually provided by Tavily

    # Content/snippet/summary directly from Tavily's processing.
    # This is typically more than just a raw snippet.
    content_summary: Optional[str] = None

    score: Optional[float] = None # Relevance score, if provided by Tavily or other search service

    # For storing full raw HTML content if fetched separately after Tavily search,
    # or if Tavily's 'include_raw_content' is used.
    raw_content_full: Optional[str] = None

    # Store any additional metadata from the search provider (e.g., Tavily)
    # This allows flexibility without cluttering the main fields.
    provider_metadata: Dict[str, Any] = field(default_factory=dict)

    # Fields from the old SearchResult that might be less relevant if Tavily is primary:
    # matched_keyword: Optional[str] = None (Tavily matches semantically, not just on one keyword)
    # snippet: Optional[str] = None (Replaced by content_summary from Tavily)


@dataclass
class TavilySearchSessionReport: # New dataclass for overall Tavily search info
    '''Holds overall information about a Tavily search session.'''
    query_echo: str # The query string sent to Tavily
    num_results_returned: int
    # Tavily might also return an overall answer or summary for the query
    overall_answer: Optional[str] = None
    # Any other session-level metadata from Tavily
    session_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessedResult: # This might still be used after our own relevance scoring/processing
    '''
    Represents a search result after local processing, potentially including
    full content fetching and local relevance re-scoring.
    '''
    processed_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    # This would now likely wrap a WebSearchResult or use its fields
    source_web_search_result: WebSearchResult

    # Content used for local relevance scoring, could be Tavily's summary or full raw_content_full
    content_used_for_scoring: Optional[str] = None

    local_relevance_score: Optional[float] = None # Our own relevance score if we re-evaluate
    local_relevance_justification: Optional[str] = None

    error_processing: Optional[str] = None # Errors during local processing (e.g. fetching full content)


@dataclass
class KnowledgeChunk: # No changes needed for this typically
    '''Represents a piece of synthesized information derived from research.'''
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    source_result_ids: List[str] = field(default_factory=list) # IDs of WebSearchResult or ProcessedResult
    confidence_score: Optional[float] = None
