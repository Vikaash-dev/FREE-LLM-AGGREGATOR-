import structlog
from typing import List, Dict, Any, Optional
import json

# Assuming LLMAggregator and related models are imported correctly if not already
from ..aggregator import LLMAggregator # If it's in src/core/aggregator.py
from ..models import ChatCompletionRequest, Message as OpenHandsMessage # If in src/models.py
from ..research_structures import SearchResult, ProcessedResult


logger = structlog.get_logger(__name__)

class ContextualKeywordExtractor:
    '''
    Extracts relevant keywords from task descriptions and context for research.
    '''
    def __init__(self, llm_aggregator: LLMAggregator): # Ensure correct type hint
        self.llm_aggregator = llm_aggregator
        logger.info("ContextualKeywordExtractor initialized.")

    async def extract(self, task_description: str, project_context_summary: Optional[str] = None) -> List[str]:
        '''
        Uses an LLM to extract relevant search keywords/phrases from the task description and project context.

        Args:
            task_description: The description of the task requiring research.
            project_context_summary: An optional summary of the project context.

        Returns:
            A list of extracted keywords/phrases. Returns an empty list if extraction fails.
        '''
        logger.info("Extracting keywords for research", task_description_preview=task_description[:50], has_project_context=bool(project_context_summary))

        context_info = ""
        if project_context_summary:
            context_info = f"Relevant Project Context:\n{project_context_summary}\n"

        prompt_template = f"""
        You are an AI assistant specialized in identifying key concepts for research.
        Based on the following task description and project context, please extract a list
        of 3 to 5 highly relevant and concise search keywords or keyphrases.
        These keywords should be suitable for use in a web search engine to find information
        that will help accomplish the task. Focus on nouns, technical terms, and specific entities.
        Avoid overly generic terms.

        Task Description:
        {task_description}

        {context_info}
        Provide the output as a JSON list of strings. For example:
        ["keyword1", "keyphrase about topic A", "specific tool name"]

        Output ONLY the JSON list. Do not include any other text.
        """

        messages = [
            OpenHandsMessage(role="system", content="You are an expert keyword extraction AI. Your output must be a valid JSON list of strings."),
            OpenHandsMessage(role="user", content=prompt_template)
        ]

        request = ChatCompletionRequest(messages=messages, model="auto", temperature=0.2) # type: ignore

        llm_response_str: Optional[str] = None
        try:
            logger.debug("Sending keyword extraction request to LLM", task_description_preview=task_description[:50])
            response = await self.llm_aggregator.chat_completion(request)

            if response.choices and response.choices[0].message and response.choices[0].message.content:
                llm_response_str = response.choices[0].message.content.strip()
                logger.debug("Received LLM response for keyword extraction", response_content_length=len(llm_response_str))

                if llm_response_str.startswith("```json"):
                    llm_response_str = llm_response_str[len("```json"):]
                if llm_response_str.endswith("```"):
                    llm_response_str = llm_response_str[:-len("```")]

                extracted_keywords = json.loads(llm_response_str)
                if isinstance(extracted_keywords, list) and all(isinstance(kw, str) for kw in extracted_keywords):
                    logger.info("Successfully extracted keywords", keywords=extracted_keywords, num_keywords=len(extracted_keywords))
                    return extracted_keywords
                else:
                    logger.warn("LLM output for keyword extraction was not a list of strings.", parsed_output_type=type(extracted_keywords).__name__, parsed_output_preview=str(extracted_keywords)[:100])
                    return [] # Fallback to empty list

            else:
                logger.warn("LLM response for keyword extraction was empty or malformed.", llm_response_obj=response.model_dump_json() if response else None)
                return []

        except json.JSONDecodeError as e:
            logger.error("Failed to decode JSON from LLM response for keyword extraction",
                         error=str(e), raw_response_snippet=llm_response_str[:200] if llm_response_str else "None")
            return []
        except Exception as e:
            logger.error("Error during LLM call for keyword extraction", error=str(e), exc_info=True)
            return []


class WebResearcher:
    '''
    Performs web searches (mocked in early Phase 2) and fetches content.
    '''
    def __init__(self):
        # Define a simple mock database of web content
        # Keys are identifiers, values are tuples of (title, content_snippet, full_content)
        # Keywords can be associated with these identifiers for mock searching.
        self.mock_web_content_db: Dict[str, Dict[str, Any]] = { # Allow Any for "keywords" list
            "mock_page_python_basics": {
                "title": "Python Basics for Beginners",
                "snippet": "Learn the fundamental concepts of Python programming...",
                "full_content": "This is a detailed document about Python basics. It covers variables, data types, loops, and functions. Suitable for beginners.",
                "keywords": ["python basics", "learn python", "python tutorial"]
            },
            "mock_page_async_python": {
                "title": "Asynchronous Programming in Python",
                "snippet": "An introduction to asyncio, async/await keywords...",
                "full_content": "Detailed guide on asynchronous programming in Python using asyncio. Discusses event loops, coroutines, async and await. For intermediate users.",
                "keywords": ["python asyncio", "async python", "python concurrency"]
            },
            "mock_page_fastapi_intro": {
                "title": "Introduction to FastAPI",
                "snippet": "Build fast APIs with Python 3.7+ based on standard Python type hints...",
                "full_content": "FastAPI is a modern, fast (high-performance) web framework for building APIs with Python. This document provides an introduction and a simple tutorial.",
                "keywords": ["fastapi tutorial", "python web framework", "fastapi basics"]
            },
            "mock_page_rust_ownership": {
                "title": "Understanding Ownership in Rust",
                "snippet": "Rust's ownership system is its most unique feature...",
                "full_content": "Ownership is a set of rules that governs how a Rust program manages memory. This page explains the concepts of ownership, borrowing, and slices.",
                "keywords": ["rust ownership", "rust memory management", "learn rust"]
            }
        }
        logger.info("WebResearcher initialized (using mock content database).")

    async def search(self, keywords: List[str]) -> List[SearchResult]:
        '''
        Simulates finding relevant mock content based on keywords.
        Does NOT perform live web searches in this version.

        Args:
            keywords: A list of keywords to "search" for.

        Returns:
            A list of SearchResult objects linking keywords to mock content identifiers.
        '''
        logger.info("Performing mock search", keywords=keywords)
        found_results: List[SearchResult] = []
        # Keep track of added identifiers to avoid duplicates for different keywords
        # pointing to the same mock page.
        added_identifiers = set()

        if not keywords:
            logger.warn("No keywords provided for mock search.")
            return []

        for keyword in keywords:
            kw_lower = keyword.lower()
            for identifier, data in self.mock_web_content_db.items():
                if identifier not in added_identifiers:
                    # Check if keyword is in the page's associated keywords or title
                    page_keywords_lower = [pk.lower() for pk in data.get("keywords", [])]
                    title_lower = data.get("title", "").lower()
                    content_snippet_lower = data.get("snippet", "").lower() # Check snippet too

                    if kw_lower in page_keywords_lower or kw_lower in title_lower or kw_lower in content_snippet_lower:
                        found_results.append(SearchResult(
                            source_identifier=identifier,
                            matched_keyword=keyword,
                            title=data.get("title"),
                            snippet=data.get("snippet")
                        ))
                        added_identifiers.add(identifier)
                        # Limit to one match per mock page for simplicity in this mock search
                        # In a real search, a page might match multiple query keywords.

        logger.info("Mock search complete", num_results_found=len(found_results), keywords=keywords)
        return found_results

    async def _fetch_mock_content(self, identifier: str) -> Optional[str]:
        '''
        Retrieves the full content for a given mock content identifier.

        Args:
            identifier: The ID of the mock content to retrieve.

        Returns:
            The full content string, or None if the identifier is not found.
        '''
        logger.debug("Fetching mock content", identifier=identifier)
        page_data = self.mock_web_content_db.get(identifier)
        if page_data:
            content = page_data.get("full_content")
            if content:
                logger.info("Mock content fetched successfully", identifier=identifier, content_length=len(content))
                return content
            else:
                logger.warn("Mock content found for identifier, but 'full_content' is missing.", identifier=identifier)
                return None
        else:
            logger.warn("Mock content identifier not found in database.", identifier=identifier)
            return None

class RelevanceScorer:
    '''
    Scores the relevance of fetched content against a task description using an LLM.
    '''
    def __init__(self, llm_aggregator: LLMAggregator): # Ensure correct type hint
        self.llm_aggregator = llm_aggregator
        logger.info("RelevanceScorer initialized.")

    async def score(self, search_results: List[SearchResult], task_description: str, web_researcher: WebResearcher) -> List[ProcessedResult]:
        '''
        Scores the relevance of content from SearchResults against the task_description using an LLM.

        Args:
            search_results: A list of SearchResult objects from the WebResearcher.
            task_description: The description of the task requiring research.
            web_researcher: An instance of WebResearcher to fetch content.

        Returns:
            A list of ProcessedResult objects with relevance scores and justifications.
        '''
        logger.info("Scoring relevance of search results", num_results=len(search_results), task_description_preview=task_description[:50])
        processed_results: List[ProcessedResult] = []

        if not search_results:
            logger.warn("No search results provided to score.")
            return []

        for s_result in search_results:
            logger.debug("Processing search result for scoring", source_identifier=s_result.source_identifier, matched_keyword=s_result.matched_keyword)
            # Use the passed web_researcher instance to fetch content
            fetched_content = await web_researcher._fetch_mock_content(s_result.source_identifier)

            if fetched_content is None:
                logger.warn("Failed to fetch content for scoring", source_identifier=s_result.source_identifier)
                processed_results.append(ProcessedResult(
                    original_search_result=s_result,
                    fetched_content="",
                    relevance_score=0.0,
                    relevance_justification="Failed to fetch content.",
                    error_fetching="Content not found or fetch error."
                ))
                continue

            # Truncate fetched_content if too long for the prompt to avoid excessive token usage
            max_content_words = 1500
            content_words = fetched_content.split()
            if len(content_words) > max_content_words:
                truncated_content = " ".join(content_words[:max_content_words]) + "..."
                logger.debug("Content truncated for LLM prompt", original_length_words=len(content_words), truncated_length_chars=len(truncated_content), source_identifier=s_result.source_identifier)
            else:
                truncated_content = fetched_content

            prompt_template = f"""
            You are an AI assistant evaluating the relevance of a piece of text to a given task.

            Task Description:
            "{task_description}"

            Text Content to Evaluate (from source: {s_result.source_identifier}, title: {s_result.title}):
            --- BEGIN CONTENT ---
            {truncated_content}
            --- END CONTENT ---

            Based on the Task Description and the Text Content, please:
            1. Provide a relevance_score (float between 0.0 for not relevant and 1.0 for highly relevant).
            2. Provide a brief justification (string, 1-2 sentences) for your score.

            Output ONLY a JSON object with the following keys: "relevance_score" and "justification".
            Example:
            {{
                "relevance_score": 0.85,
                "justification": "The content directly addresses key aspects of the task and provides useful information."
            }}
            """

            messages = [
                OpenHandsMessage(role="system", content="You are an expert relevance evaluation AI. Your output must be a valid JSON object as specified."),
                OpenHandsMessage(role="user", content=prompt_template)
            ]
            request = ChatCompletionRequest(messages=messages, model="auto", temperature=0.1) # type: ignore

            llm_score_json_str: Optional[str] = None
            current_score = 0.0
            current_justification = "LLM scoring failed or produced invalid output."

            try:
                logger.debug("Sending relevance scoring request to LLM", source_identifier=s_result.source_identifier)
                response = await self.llm_aggregator.chat_completion(request)

                if response.choices and response.choices[0].message and response.choices[0].message.content:
                    llm_score_json_str = response.choices[0].message.content.strip()
                    logger.debug("Received LLM response for relevance scoring", llm_response_content_length=len(llm_score_json_str), source_identifier=s_result.source_identifier)

                    if llm_score_json_str.startswith("```json"):
                        llm_score_json_str = llm_score_json_str[len("```json"):]
                    if llm_score_json_str.endswith("```"):
                        llm_score_json_str = llm_score_json_str[:-len("```")]

                    score_data = json.loads(llm_score_json_str)
                    current_score = float(score_data.get("relevance_score", 0.0))
                    current_justification = score_data.get("justification", "No justification provided by LLM.")
                    logger.info("Successfully scored relevance", source_identifier=s_result.source_identifier, score=current_score)
                else:
                    logger.warn("LLM response for relevance scoring was empty or malformed.", source_identifier=s_result.source_identifier, llm_response_obj=response.model_dump_json() if response else None)

            except json.JSONDecodeError as e:
                logger.error("Failed to decode JSON from LLM response for relevance scoring",
                             error=str(e), raw_response_snippet=llm_score_json_str[:200] if llm_score_json_str else "None", source_identifier=s_result.source_identifier)
            except ValueError as e: # Handles float conversion error
                logger.error("Failed to convert relevance_score to float from LLM response",
                             error=str(e), raw_response_snippet=llm_score_json_str[:200] if llm_score_json_str else "None", source_identifier=s_result.source_identifier)
            except Exception as e:
                logger.error("Error during LLM call for relevance scoring", error=str(e), source_identifier=s_result.source_identifier, exc_info=True)

            processed_results.append(ProcessedResult(
                original_search_result=s_result,
                fetched_content=fetched_content, # Store original full content, not truncated
                relevance_score=current_score,
                relevance_justification=current_justification
            ))

        logger.info("Relevance scoring complete for all results.", num_processed=len(processed_results), num_originally_found=len(search_results))
        return processed_results
