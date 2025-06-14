import structlog
from typing import List, Dict, Any, Optional
import json

# Assuming LLMAggregator and related models are imported correctly
from .aggregator import LLMAggregator
from ..models import ChatCompletionRequest, Message as OpenHandsMessage # Ensure OpenHandsMessage alias

# Import necessary structures
from .research_structures import ProcessedResult, KnowledgeChunk, ResearchQuery, SearchResult # Added SearchResult
from .research_components import ContextualKeywordExtractor, WebResearcher, RelevanceScorer


logger = structlog.get_logger(__name__)

class IntelligentResearchAssistant:
    '''
    Orchestrates the research process using various components like
    keyword extraction, web searching, and relevance scoring to synthesize knowledge.
    '''

    def __init__(self,
                 keyword_extractor: ContextualKeywordExtractor,
                 web_researcher: WebResearcher,
                 relevance_scorer: RelevanceScorer,
                 llm_aggregator: LLMAggregator): # LLM Aggregator for synthesis step
        '''
        Initializes the IntelligentResearchAssistant.

        Args:
            keyword_extractor: Instance of ContextualKeywordExtractor.
            web_researcher: Instance of WebResearcher.
            relevance_scorer: Instance of RelevanceScorer.
            llm_aggregator: Instance of LLMAggregator for the synthesis step.
        '''
        self.keyword_extractor = keyword_extractor
        self.web_researcher = web_researcher
        self.relevance_scorer = relevance_scorer
        self.llm_aggregator = llm_aggregator # For synthesis
        logger.info("IntelligentResearchAssistant initialized with components.")

    async def research_for_task(self, research_query: ResearchQuery) -> List[KnowledgeChunk]:
        '''
        Main orchestration method for performing research for a given task query.
        It extracts keywords, performs (mock) web searches, scores relevance,
        and synthesizes the findings into knowledge chunks.

        Args:
            research_query: A ResearchQuery object containing the task description,
                            context, and other parameters for research.

        Returns:
            A list of KnowledgeChunk objects containing synthesized information,
            or an empty list if research fails or yields no useful information.
        '''
        logger.info("Starting research for task",
                    query_id=research_query.query_id,
                    task_description_preview=research_query.original_task_description[:50])

        # Step 1: Extract Keywords
        logger.debug("Extracting keywords...", query_id=research_query.query_id)
        # Keywords were already extracted and placed in research_query if called from simulation script
        # If research_query.keywords is empty, then call extractor.
        if not research_query.keywords:
            extracted_keywords = await self.keyword_extractor.extract(
                task_description=research_query.original_task_description,
                project_context_summary=research_query.project_context_summary
            )
            if not extracted_keywords:
                logger.warn("No keywords extracted. Research cannot proceed effectively.", query_id=research_query.query_id)
                return []
            research_query.keywords = extracted_keywords

        logger.info("Keywords for research", keywords=research_query.keywords, query_id=research_query.query_id)


        # Step 2: Search (using WebResearcher with mock content)
        logger.debug("Performing (mock) web search with extracted keywords...", query_id=research_query.query_id)
        # Type hint for search_results_raw should be List[SearchResult]
        search_results_raw: List[SearchResult] = await self.web_researcher.search(research_query.keywords)
        if not search_results_raw:
            logger.warn("No search results obtained from WebResearcher.", keywords=research_query.keywords, query_id=research_query.query_id)
            return [] # If no raw results, can't proceed.
        logger.info("Mock search complete", num_raw_results=len(search_results_raw), query_id=research_query.query_id)

        # Step 3: Score Relevance
        logger.debug("Scoring relevance of search results...", query_id=research_query.query_id)
        # Type hint for processed_results should be List[ProcessedResult]
        processed_results: List[ProcessedResult] = await self.relevance_scorer.score(
            search_results_raw,
            research_query.original_task_description,
            self.web_researcher # Pass the web_researcher instance
        )
        # Note: relevance_scorer.score might return empty list if all content fetching failed or other issues.
        # We don't necessarily stop if processed_results is empty, as synthesize_research can handle it.

        # Filter for relevant results before synthesis
        relevant_results = [res for res in processed_results if res.relevance_score >= 0.5 and not res.error_fetching]
        if not relevant_results:
            logger.warn("No sufficiently relevant results after scoring for synthesis (or all failed fetching).", query_id=research_query.query_id, num_processed_results=len(processed_results))
            # Return an empty list or a KnowledgeChunk indicating no relevant info found.
            # For consistency with synthesize_research, let it decide.
            # However, if all failed fetching, it's good to know.
            if processed_results and all(res.error_fetching for res in processed_results):
                 return [KnowledgeChunk(content=f"Failed to fetch content for all {len(processed_results)} search results for query: {research_query.original_task_description[:30]}...", source_result_ids=[])]


        logger.info("Relevance scoring complete", num_processed_results=len(processed_results), num_relevant_for_synthesis=len(relevant_results), query_id=research_query.query_id)


        # Step 4: Synthesize Knowledge
        logger.debug("Synthesizing knowledge from relevant results...", query_id=research_query.query_id)
        synthesized_knowledge: List[KnowledgeChunk] = await self.synthesize_research(
            relevant_results, # Pass only relevant results
            original_query=research_query
        )
        # synthesize_research handles empty relevant_results and may return a specific chunk.

        if not synthesized_knowledge or (len(synthesized_knowledge) == 1 and "No specific information found" in synthesized_knowledge[0].content) :
            logger.warn("Knowledge synthesis yielded no substantial chunks or only an 'info not found' chunk.", query_id=research_query.query_id)
            # We might still return the "info not found" chunk.

        logger.info("Research for task complete. Knowledge chunks synthesized.",
                    num_chunks=len(synthesized_knowledge), query_id=research_query.query_id)
        return synthesized_knowledge

    async def synthesize_research(self, processed_results: List[ProcessedResult], original_query: Optional[ResearchQuery] = None, max_results_to_synthesize: int = 3) -> List[KnowledgeChunk]:
        '''
        Synthesizes information from processed research results into KnowledgeChunks using an LLM.

        Args:
            processed_results: A list of ProcessedResult objects, typically sorted by relevance.
            original_query: The original ResearchQuery that led to these results, for context.
            max_results_to_synthesize: The maximum number of top results to use for synthesis.

        Returns:
            A list of KnowledgeChunk objects containing synthesized information.
        '''
        logger.info("Synthesizing research from processed results", num_results=len(processed_results), max_to_use=max_results_to_synthesize)

        if not processed_results:
            logger.warn("No processed results to synthesize.")
            return []

        # Sort by relevance score (descending) and select top N
        # Assuming ProcessedResult has a 'relevance_score' attribute
        sorted_results = sorted(processed_results, key=lambda r: r.relevance_score, reverse=True)
        top_results = sorted_results[:max_results_to_synthesize]

        if not top_results:
            logger.warn("No results selected for synthesis after sorting/filtering.")
            return []

        # Prepare content for the LLM prompt
        content_for_synthesis = ""
        source_ids_used = []
        for i, result in enumerate(top_results):
            if result.fetched_content and result.relevance_score > 0.3: # Basic threshold
                # Truncate individual content pieces if they are too long
                max_words_per_source = 1000
                words = result.fetched_content.split()
                if len(words) > max_words_per_source:
                    content_to_add = " ".join(words[:max_words_per_source]) + "..."
                else:
                    content_to_add = result.fetched_content

                content_for_synthesis += f"Source {i+1} (ID: {result.original_search_result.source_identifier}, Relevance: {result.relevance_score:.2f}):\n"
                content_for_synthesis += f"{content_to_add}\n\n---\n\n"
                source_ids_used.append(result.processed_id) # Use processed_id as it's unique for this processing run
            else:
                 logger.debug("Skipping result in synthesis due to low relevance or no content", result_id=result.processed_id if hasattr(result, 'processed_id') else 'N/A')


        if not content_for_synthesis:
            logger.warn("No content suitable for synthesis after filtering top results.")
            return []

        task_context_prompt = ""
        if original_query:
            task_context_prompt = f"The original research was for the task: '{original_query.original_task_description}'. Keywords used: {original_query.keywords}."

        prompt_template = f"""
        You are an AI assistant that synthesizes information from multiple text sources into concise, useful knowledge.
        {task_context_prompt}

        Below are several pieces of text content obtained from research.
        Your task is to:
        1. Read and understand all provided text content.
        2. Identify the most important and relevant pieces of information related to the research context (if provided).
        3. Synthesize this information into a list of distinct, concise, and factual "knowledge chunks". Each chunk should represent a single key point or a coherent piece of information.
        4. Aim for 3-5 high-quality knowledge chunks.

        Provided Text Content:
        --- BEGIN CONTENT ---
        {content_for_synthesis}
        --- END CONTENT ---

        Output ONLY a JSON list of strings, where each string is a knowledge chunk.
        Example:
        [
            "Knowledge chunk 1 summarizing a key finding.",
            "Knowledge chunk 2 detailing another important point.",
            "Knowledge chunk 3 presenting a relevant fact or data."
        ]
        """
        messages = [
            OpenHandsMessage(role="system", content="You are an expert information synthesis AI. Your output must be a valid JSON list of strings, each representing a knowledge chunk."),
            OpenHandsMessage(role="user", content=prompt_template)
        ]
        request = ChatCompletionRequest(messages=messages, model="auto", temperature=0.3) # type: ignore

        llm_synthesis_json_str: Optional[str] = None
        try:
            logger.debug("Sending synthesis request to LLM", num_sources=len(top_results))
            response = await self.llm_aggregator.chat_completion(request)

            if response.choices and response.choices[0].message and response.choices[0].message.content:
                llm_synthesis_json_str = response.choices[0].message.content.strip()
                logger.debug("Received LLM response for synthesis", llm_response_content_length=len(llm_synthesis_json_str))

                if llm_synthesis_json_str.startswith("```json"):
                    llm_synthesis_json_str = llm_synthesis_json_str[len("```json"):]
                if llm_synthesis_json_str.endswith("```"):
                    llm_synthesis_json_str = llm_synthesis_json_str[:-len("```")]

                synthesized_strings = json.loads(llm_synthesis_json_str)

                knowledge_chunks: List[KnowledgeChunk] = []
                if isinstance(synthesized_strings, list) and all(isinstance(s, str) for s in synthesized_strings):
                    for s_content in synthesized_strings:
                        knowledge_chunks.append(KnowledgeChunk(
                            content=s_content,
                            source_result_ids=source_ids_used # Associate all used sources with each chunk for simplicity
                            # TODO: Could try to get LLM to map chunks to specific source IDs if needed
                        ))
                    logger.info("Successfully synthesized knowledge chunks", num_chunks=len(knowledge_chunks))
                    return knowledge_chunks
                else:
                    logger.warn("LLM output for synthesis was not a list of strings.", parsed_output_type=type(synthesized_strings).__name__, parsed_output_preview=str(synthesized_strings)[:100])
                    return []
            else:
                logger.warn("LLM response for synthesis was empty or malformed.", llm_response_obj=response.model_dump_json() if response else None)
                return []

        except json.JSONDecodeError as e:
            logger.error("Failed to decode JSON from LLM response for synthesis",
                         error=str(e), raw_response_snippet=llm_synthesis_json_str[:200] if llm_synthesis_json_str else "None")
            return []
        except Exception as e:
            logger.error("Error during LLM call for synthesis", error=str(e), exc_info=True)
            return []
