import structlog
from typing import List, Dict, Any, Optional
import json
import httpx # Import httpx
from bs4 import BeautifulSoup # Import BeautifulSoup
from tavily import TavilyClient # Import TavilyClient
import asyncio # For asyncio.to_thread

# Assuming LLMAggregator and related models are imported correctly if not already
from ..aggregator import LLMAggregator # If it's in src/core/aggregator.py
from ..models import ChatCompletionRequest, Message as OpenHandsMessage # If in src/models.py
# Import updated data structures
from ..research_structures import WebSearchResult, TavilySearchSessionReport, ProcessedResult


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
    Performs web searches using Tavily AI as the primary engine and can fetch live content.
    The old Google scraping search is kept as a deprecated internal method.
    '''
    def __init__(self,
                 tavily_api_key: str, # Changed: Tavily API key is now mandatory
                 http_client: Optional[httpx.AsyncClient] = None):
        '''
        Initializes the WebResearcher.

        Args:
            tavily_api_key: The API key for Tavily AI.
            http_client: Optional shared httpx.AsyncClient. A new one is created if None.
        '''
        if not tavily_api_key:
            logger.error("Tavily API key not provided to WebResearcher. Tavily search will fail.")
            # Or raise ValueError("Tavily API key is required.")
            # For now, allow initialization but log error. Calls will fail.
        self.tavily_client = TavilyClient(api_key=tavily_api_key)
        self.http_client = http_client if http_client else httpx.AsyncClient(timeout=10.0, follow_redirects=True)
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        logger.info("WebResearcher initialized with TavilyClient.")

    async def _search_google_scrape(self, keywords: List[str], max_results_per_keyword: int = 1) -> List[WebSearchResult]:
        '''
        DEPRECATED: Basic web search by constructing a Google search URL and parsing results.
        This is fragile and kept for fallback/comparison only.
        '''
        # ... (Implementation from previous version can be kept here, but marked as deprecated) ...
        # For brevity in this subtask, let's assume it's minimized or just logs a deprecation warning.
        logger.warn("_search_google_scrape is deprecated and should not be primarily used. Returning empty list.", keywords=keywords)
        return []


    async def search_with_tavily(self,
                                 query: str,
                                 search_depth: str = "basic", # "basic" or "advanced"
                                 max_results: int = 5,
                                 include_answer: bool = False,
                                 # include_raw_content: bool = False, # Tavily's raw_content is different
                                 include_domains: Optional[List[str]] = None,
                                 exclude_domains: Optional[List[str]] = None
                                 ) -> Dict[str, Any]: # Returns a dict: {"report": TavilySearchSessionReport, "results": List[WebSearchResult]}
        '''
        Performs a search using the Tavily AI API.

        Args:
            query: The search query string (natural language or keywords).
            search_depth: Tavily search depth ("basic" or "advanced").
            max_results: Maximum number of search results to return.
            include_answer: Whether Tavily should include an overall answer to the query.
            # include_raw_content: Whether Tavily should include raw content of visited pages.
            include_domains: Optional list of domains to focus search on.
            exclude_domains: Optional list of domains to exclude from search.

        Returns:
            A dictionary containing a 'report' (TavilySearchSessionReport) and
            'results' (List[WebSearchResult]). Returns empty list of results on failure.
        '''
        logger.info("Performing search with Tavily AI", query=query, search_depth=search_depth, max_results=max_results)

        output_results: List[WebSearchResult] = []
        session_report_data: Dict[str, Any] = {
            "query_echo": query,
            "num_results_returned": 0,
            "overall_answer": None,
            "session_metadata": {}
        }

        if not self.tavily_client.api_key: # Check if API key was actually set
            logger.error("Tavily API key is not configured in WebResearcher. Cannot perform search.")
            return {"report": TavilySearchSessionReport(**session_report_data), "results": output_results}

        try:
            # Use asyncio.to_thread to run the blocking TavilyClient.search call
            response_data = await asyncio.to_thread(
                self.tavily_client.search,
                query=query,
                search_depth=search_depth,
                max_results=max_results,
                include_answer=include_answer,
                include_domains=include_domains,
                exclude_domains=exclude_domains
            )

            logger.debug("Received response from Tavily API", tavily_response_keys=list(response_data.keys()))

            session_report_data["overall_answer"] = response_data.get("answer")
            session_report_data["session_metadata"] = {k: v for k, v in response_data.items() if k not in ["results", "answer", "query"]}


            if "results" in response_data and isinstance(response_data["results"], list):
                session_report_data["num_results_returned"] = len(response_data["results"])
                for tavily_result in response_data["results"]:
                    wsr = WebSearchResult(
                        url=tavily_result.get("url", ""),
                        title=tavily_result.get("title"),
                        content_summary=tavily_result.get("content"), # Tavily's 'content' is usually a summary/snippet
                        score=float(tavily_result.get("score", 0.0)) if tavily_result.get("score") else None,
                        # raw_content_full would be populated if we fetch it ourselves later
                        provider_metadata={"tavily_result": tavily_result} # Store the whole Tavily item for reference
                    )
                    output_results.append(wsr)
                logger.info("Successfully processed Tavily search results.", num_results=len(output_results))
            else:
                logger.warn("Tavily API response did not contain 'results' list or it was malformed.", tavily_response=response_data)

        except Exception as e:
            logger.error("Error during Tavily API call", query=query, error=str(e), exc_info=True)
            session_report_data["session_metadata"]["error"] = str(e)
            # Return empty results but with the report indicating failure context

        return {
            "report": TavilySearchSessionReport(**session_report_data),
            "results": output_results
        }


    async def fetch_and_parse_live_content(self, url: str) -> Optional[str]:
        '''
        Fetches content from a live URL and parses it to extract clean text.

        Args:
            url: The URL to fetch and parse.

        Returns:
            Cleaned textual content as a string, or None if fetching/parsing fails.
        '''
        logger.info("Fetching live content", url=url)
        try:
            response = await self.http_client.get(url, headers=self.headers)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "").lower()
            if not ("html" in content_type or "text" in content_type):
                logger.warn("Skipping non-HTML/text content", url=url, content_type=content_type)
                return None

            soup = BeautifulSoup(response.text, 'html.parser')

            for script_or_style in soup(["script", "style", "header", "footer", "nav", "aside"]):
                script_or_style.decompose()

            text = soup.get_text(separator='\n', strip=True)
            text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])

            logger.info("Live content fetched and parsed successfully", url=url, content_length=len(text))
            return text

        except httpx.HTTPStatusError as e:
            logger.error("HTTP error fetching live content", url=e.request.url, status_code=e.response.status_code, exc_info=False)
        except httpx.RequestError as e:
            logger.error("Request error fetching live content", url=e.request.url, error=str(e), exc_info=True)
        except Exception as e:
            logger.error("Unexpected error fetching/parsing live content", url=url, error=str(e), exc_info=True)

        return None

    async def close_client(self):
        '''Closes the httpx.AsyncClient. Should be called on application shutdown.'''
        if self.http_client:
            await self.http_client.aclose()
            logger.info("WebResearcher's HTTP client closed.")

class RelevanceScorer:
    '''
    Scores the relevance of fetched content against a task description using an LLM.
    '''
    def __init__(self, llm_aggregator: LLMAggregator): # Ensure correct type hint
        self.llm_aggregator = llm_aggregator
        logger.info("RelevanceScorer initialized.")

    async def score(self,
                    web_search_results: List[WebSearchResult],
                    task_description: str,
                    web_researcher: WebResearcher,
                    fetch_full_content_threshold: Optional[float] = 0.6, # Score below which we try to fetch full content
                    min_summary_length_for_direct_use: int = 50 # Min length of Tavily summary to use it directly
                   ) -> List[ProcessedResult]:
        '''
        Scores the relevance of content from WebSearchResults against the task_description using an LLM.
        Uses the provided WebResearcher instance to fetch live content if Tavily's summary is insufficient.

        Args:
            search_results: A list of SearchResult objects from the WebResearcher.
            task_description: The description of the task requiring research.
            web_researcher: An instance of WebResearcher to fetch content.

        Returns:
            A list of ProcessedResult objects with relevance scores and justifications.
        '''
        logger.info("Starting local relevance scoring of web search results", num_results=len(web_search_results), task_description_preview=task_description[:100])
        processed_outputs: List[ProcessedResult] = []

        if not web_search_results:
            logger.warn("No web search results provided to RelevanceScorer.")
            return []

        for ws_result in web_search_results:
            logger.debug("Processing WebSearchResult for local scoring", url=ws_result.url, tavily_score=ws_result.score)

            content_to_evaluate: Optional[str] = None
            content_source_type = "unknown"
            error_processing: Optional[str] = None # Changed from error_fetching_message

            # Determine if Tavily's summary is sufficient or if full content is needed
            use_tavily_summary = ws_result.content_summary and len(ws_result.content_summary) >= min_summary_length_for_direct_use

            should_fetch_full = False
            if not use_tavily_summary: # If summary is too short or missing
                logger.info("Tavily summary is short or missing. Will attempt to fetch full content.", url=ws_result.url, summary_len=len(ws_result.content_summary or ""))
                should_fetch_full = True
            # Or if Tavily's score is below a threshold (and threshold is set)
            elif ws_result.score is not None and fetch_full_content_threshold is not None and ws_result.score < fetch_full_content_threshold:
                logger.info("Tavily score below threshold. Will attempt to fetch full content for re-evaluation.", url=ws_result.url, tavily_score=ws_result.score)
                should_fetch_full = True

            if should_fetch_full:
                if ws_result.raw_content_full: # If full content was already part of WebSearchResult (e.g., from Tavily's include_raw_content)
                    content_to_evaluate = ws_result.raw_content_full
                    content_source_type = "pre_fetched_raw_content_from_websearchresult"
                    logger.debug("Using pre-fetched raw_content_full from WebSearchResult.", url=ws_result.url)
                else: # Fetch it now
                    logger.debug("Fetching full live content for local scoring.", url=ws_result.url)
                    full_content = await web_researcher.fetch_and_parse_live_content(ws_result.url)
                    if full_content:
                        content_to_evaluate = full_content
                        ws_result.raw_content_full = full_content # Store it back if fetched
                        content_source_type = "fetched_live_content_via_relevancescorer"
                    else:
                        error_processing = "Full content fetch failed or yielded no text."
                        logger.warn(error_processing, url=ws_result.url)
                        if ws_result.content_summary: # Fallback to Tavily summary if fetch failed but summary exists
                            content_to_evaluate = ws_result.content_summary
                            content_source_type = "tavily_summary_after_fetch_fail"
                            logger.info("Falling back to Tavily summary as full content fetch failed.", url=ws_result.url)
                        else: # No summary and fetch failed
                            logger.error("No content available for scoring (no summary, and full content fetch failed).", url=ws_result.url)
                            processed_outputs.append(ProcessedResult(
                                source_web_search_result=ws_result, content_used_for_scoring=None,
                                local_relevance_score=0.0, local_relevance_justification="No content available for local scoring.",
                                error_processing=error_processing or "No content available."
                            ))
                            continue # Next search result
            elif use_tavily_summary: # Use Tavily's summary directly
                content_to_evaluate = ws_result.content_summary
                content_source_type = "tavily_summary_direct"
                logger.debug("Using Tavily's content_summary directly for local scoring.", url=ws_result.url)

            if not content_to_evaluate: # Should only happen if all attempts to get content failed
                 logger.error("Logic error: No content_to_evaluate was set for LLM scoring.", url=ws_result.url)
                 processed_outputs.append(ProcessedResult(
                    source_web_search_result=ws_result, content_used_for_scoring=None,
                    local_relevance_score=0.0, local_relevance_justification="Internal error: No content selected for evaluation.",
                    error_processing=error_processing or "Internal error selecting content for evaluation."
                 ))
                 continue # Next search result

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

            Text Content to Evaluate (from URL: {ws_result.url}, Original Score: {ws_result.score if ws_result.score is not None else 'N/A'}, Source Type: {content_source_type}):
            --- BEGIN CONTENT ---
            {truncated_content}
            --- END CONTENT ---

            Based on the Task Description and the Text Content, please provide:
            1. A 'local_relevance_score' (float between 0.0 for not relevant and 1.0 for highly relevant). This is your independent assessment.
            2. A brief 'local_justification' (string, 1-2 sentences) for your score.
            Output ONLY a JSON object: {{"local_relevance_score": <float>, "local_justification": "<string>"}}
            """
            messages = [OpenHandsMessage(role="system", content="You are an expert relevance evaluation AI. Your output must be a valid JSON object as specified."), OpenHandsMessage(role="user", content=prompt_template)]
            request = ChatCompletionRequest(messages=messages, model="auto", temperature=0.1) # type: ignore

            llm_score_json_str: Optional[str] = None
            current_local_score = 0.0
            current_local_justification = "LLM local scoring failed or produced invalid output."

            try:
                logger.debug("Sending local relevance scoring request to LLM", source_url=ws_result.url, content_source_type=content_source_type)
                response = await self.llm_aggregator.chat_completion(request)
                if response.choices and response.choices[0].message and response.choices[0].message.content:
                    llm_score_json_str = response.choices[0].message.content.strip()
                    if llm_score_json_str.startswith("```json"): llm_score_json_str = llm_score_json_str[len("```json"):]
                    if llm_score_json_str.endswith("```"): llm_score_json_str = llm_score_json_str[:-len("```")]
                    score_data = json.loads(llm_score_json_str)
                    current_local_score = float(score_data.get("local_relevance_score", 0.0))
                    current_local_justification = score_data.get("local_justification", "No local justification by LLM.")
                    logger.info("Successfully performed local relevance scoring", url=ws_result.url, local_score=current_local_score)
                else:
                    logger.warn("LLM response for local scoring was empty/malformed.", source_url=ws_result.url, llm_response_obj=response.model_dump_json() if response else None)
            except (json.JSONDecodeError, ValueError) as e_parse:
                logger.error("Failed to parse/convert LLM response for local scoring", error=str(e_parse), raw_response_snippet=llm_score_json_str[:200] if llm_score_json_str else "None", source_url=ws_result.url)
                error_processing = (error_processing + "; " if error_processing else "") + f"LLM response parsing error: {str(e_parse)}"
            except Exception as e_llm:
                logger.error("Error during LLM call for local relevance scoring", error=str(e_llm), source_url=ws_result.url, exc_info=True)
                error_processing = (error_processing + "; " if error_processing else "") + f"LLM call error: {str(e_llm)}"

            processed_outputs.append(ProcessedResult(
                source_web_search_result=ws_result,
                content_used_for_scoring=content_to_evaluate, # This is the content (summary or full) sent to LLM
                local_relevance_score=current_local_score,
                local_relevance_justification=current_local_justification,
                error_processing=error_processing
            ))

        logger.info("Local relevance scoring batch complete.", num_input=len(web_search_results), num_processed_successfully=len([p for p in processed_outputs if not p.error_processing]))
        return processed_outputs
