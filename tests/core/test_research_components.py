import unittest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import httpx # For httpx.Response and potential errors
from tavily import TavilyClient # Import TavilyClient

from src.core.research_components import ContextualKeywordExtractor, WebResearcher, RelevanceScorer
# Import updated data structures
from src.core.research_structures import WebSearchResult, ProcessedResult, TavilySearchSessionReport, ResearchQuery
from src.models import ChatCompletionResponse, Choice, Message as OpenHandsMessage
# Assuming LLMAggregator is needed for ContextualKeywordExtractor and RelevanceScorer
from src.core.aggregator import LLMAggregator
# Need IntelligentResearchAssistant for the new test class
from src.core.researcher import IntelligentResearchAssistant
from src.core.research_structures import KnowledgeChunk


class TestContextualKeywordExtractor(unittest.TestCase):
    def setUp(self):
        self.mock_llm_aggregator = AsyncMock(spec=LLMAggregator)
        self.keyword_extractor = ContextualKeywordExtractor(llm_aggregator=self.mock_llm_aggregator)

    def _run_async(self, coro):
        return asyncio.run(coro)

    def test_extract_keywords_success(self):
        task_desc = "Analyze Python asyncio best practices for web servers."
        context_summary = "Project focuses on high-performance http services."
        mock_response_content = '["python asyncio", "web server performance", "async best practices"]'

        mock_llm_choice = Choice(index=0, message=OpenHandsMessage(role="assistant", content=mock_response_content), finish_reason="stop")
        mock_llm_response = ChatCompletionResponse(id="kw_resp_1", object="chat.completion", created=0, model="sim_model", choices=[mock_llm_choice])
        self.mock_llm_aggregator.chat_completion = AsyncMock(return_value=mock_llm_response)

        extracted_keywords = self._run_async(self.keyword_extractor.extract(task_desc, context_summary))

        self.mock_llm_aggregator.chat_completion.assert_called_once()
        self.assertEqual(extracted_keywords, ["python asyncio", "web server performance", "async best practices"])

    def test_extract_keywords_llm_json_error(self):
        task_desc = "Research quantum computing."
        mock_response_content = 'This is not JSON'
        mock_llm_choice = Choice(index=0, message=OpenHandsMessage(role="assistant", content=mock_response_content), finish_reason="stop")
        mock_llm_response = ChatCompletionResponse(id="kw_resp_2", object="chat.completion", created=0, model="sim_model", choices=[mock_llm_choice])
        self.mock_llm_aggregator.chat_completion = AsyncMock(return_value=mock_llm_response)

        extracted_keywords = self._run_async(self.keyword_extractor.extract(task_desc, None))
        self.assertEqual(extracted_keywords, []) # Should fallback to empty list

    def test_extract_keywords_llm_returns_not_list(self):
        task_desc = "Research LLM agents."
        mock_response_content = '{"keywords": "not a list"}' # valid JSON but not a list
        mock_llm_choice = Choice(index=0, message=OpenHandsMessage(role="assistant", content=mock_response_content), finish_reason="stop")
        mock_llm_response = ChatCompletionResponse(id="kw_resp_3", object="chat.completion", created=0, model="sim_model", choices=[mock_llm_choice])
        self.mock_llm_aggregator.chat_completion = AsyncMock(return_value=mock_llm_response)

        extracted_keywords = self._run_async(self.keyword_extractor.extract(task_desc, None))
        self.assertEqual(extracted_keywords, [])

    def test_extract_keywords_llm_empty_response(self):
        task_desc = "Research AGI."
        mock_llm_choice = Choice(index=0, message=OpenHandsMessage(role="assistant", content=""), finish_reason="stop") # Empty content
        mock_llm_response = ChatCompletionResponse(id="kw_resp_4", object="chat.completion", created=0, model="sim_model", choices=[mock_llm_choice])
        self.mock_llm_aggregator.chat_completion = AsyncMock(return_value=mock_llm_response)

        extracted_keywords = self._run_async(self.keyword_extractor.extract(task_desc, None))
        self.assertEqual(extracted_keywords, [])


class TestWebResearcher(unittest.TestCase):
    def setUp(self):
        self.mock_tavily_client = MagicMock(spec=TavilyClient)
        self.mock_http_client = AsyncMock(spec=httpx.AsyncClient) # For fetch_and_parse_live_content
        self.web_researcher = WebResearcher(tavily_api_key="test_tavily_key", http_client=self.mock_http_client)
        # Patch the TavilyClient instance within the WebResearcher instance
        self.web_researcher.tavily_client = self.mock_tavily_client

    def _run_async(self, coro):
        return asyncio.run(coro)

    def _prepare_mock_response(self, text_content: str, status_code: int = 200, headers: Optional[dict] = None):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.text = text_content
        mock_response.status_code = status_code
        mock_response.headers = headers or {"content-type": "text/html"}

        if status_code >= 400:
            mock_response.raise_for_status = MagicMock(side_effect=httpx.HTTPStatusError("Error", request=MagicMock(), response=mock_response))
        else:
            def pass_mock(): pass
            mock_response.raise_for_status = MagicMock(side_effect=pass_mock)
        return mock_response

    @unittest.skip("Google scraping search is deprecated.")
    def test_search_google_scrape_deprecated(self):
        # This test is for the old _search_google_scrape method if we want to keep it
        # For now, we'll just ensure it logs and returns empty
        keywords = ["python asyncio"]
        with patch.object(self.web_researcher, 'logger') as mock_logger:
            results = self._run_async(self.web_researcher._search_google_scrape(keywords))
            self.assertEqual(results, [])
            mock_logger.warn.assert_called_with("_search_google_scrape is deprecated and should not be primarily used. Returning empty list.", keywords=keywords)

    @patch('asyncio.to_thread') # Mock asyncio.to_thread
    def test_search_with_tavily_success(self, mock_to_thread):
        query = "python asyncio best practices"
        mock_tavily_api_response = {
            "query": query,
            "answer": "Tavily's summarized answer about asyncio.",
            "results": [
                {"title": "Async IO in Python", "url": "https://example.com/asyncio", "content": "A guide to Python's asyncio.", "score": 0.95},
                {"title": "Python Concurrency", "url": "https://example.com/concurrency", "content": "Understanding concurrency in Python.", "score": 0.90}
            ],
            "response_time": 0.5
        }
        # Configure mock_to_thread to return a future that resolves to your mock response
        mock_future = asyncio.Future()
        mock_future.set_result(mock_tavily_api_response)
        mock_to_thread.return_value = mock_future
        # self.mock_tavily_client.search = MagicMock(return_value=mock_tavily_api_response) # Alternative if not mocking to_thread

        response_dict = self._run_async(self.web_researcher.search_with_tavily(query, include_answer=True))

        mock_to_thread.assert_called_once_with(
            self.mock_tavily_client.search, # The function that was supposed to be called in a thread
            query=query,
            search_depth="basic",
            max_results=5,
            include_answer=True,
            include_domains=None,
            exclude_domains=None
        )

        self.assertIsInstance(response_dict["report"], TavilySearchSessionReport)
        self.assertEqual(response_dict["report"].overall_answer, "Tavily's summarized answer about asyncio.")
        self.assertEqual(len(response_dict["results"]), 2)
        self.assertIsInstance(response_dict["results"][0], WebSearchResult)
        self.assertEqual(response_dict["results"][0].url, "https://example.com/asyncio")
        self.assertEqual(response_dict["results"][0].content_summary, "A guide to Python's asyncio.")
        self.assertEqual(response_dict["results"][0].score, 0.95)

    @patch('asyncio.to_thread')
    def test_search_with_tavily_no_results(self, mock_to_thread):
        query = "unknown topic"
        mock_tavily_api_response = {"query": query, "results": [], "response_time": 0.2}
        mock_future = asyncio.Future()
        mock_future.set_result(mock_tavily_api_response)
        mock_to_thread.return_value = mock_future

        response_dict = self._run_async(self.web_researcher.search_with_tavily(query))
        self.assertEqual(len(response_dict["results"]), 0)
        self.assertEqual(response_dict["report"].num_results_returned, 0)

    @patch('asyncio.to_thread')
    def test_search_with_tavily_api_error(self, mock_to_thread):
        query = "trigger error"
        mock_to_thread.side_effect = Exception("Tavily API Error")

        response_dict = self._run_async(self.web_researcher.search_with_tavily(query))
        self.assertEqual(len(response_dict["results"]), 0)
        self.assertIn("error", response_dict["report"].session_metadata)
        self.assertEqual(response_dict["report"].session_metadata["error"], "Tavily API Error")

    def test_fetch_and_parse_live_content_success(self):
        url = "https://example.com/test_page"
        html_content = "<html><head><title>Test</title><style>.hide{display:none}</style></head><body><script>alert('hi')</script><p>Main content here.</p><footer>Footer</footer></body></html>"
        self.mock_http_client.get = AsyncMock(return_value=self._prepare_mock_response(html_content))

        content = self._run_async(self.web_researcher.fetch_and_parse_live_content(url))

        self.mock_http_client.get.assert_called_once_with(url, headers=self.web_researcher.headers)
        self.assertEqual(content, "Main content here.")

    def test_fetch_and_parse_live_content_non_html(self):
        url = "https://example.com/image.jpg"
        self.mock_http_client.get = AsyncMock(return_value=self._prepare_mock_response("binary data", headers={"content-type": "image/jpeg"}))
        content = self._run_async(self.web_researcher.fetch_and_parse_live_content(url))
        self.assertIsNone(content)

    def test_fetch_and_parse_live_content_http_error(self):
        url = "https://example.com/error_page"
        self.mock_http_client.get = AsyncMock(return_value=self._prepare_mock_response("Error", status_code=404))
        content = self._run_async(self.web_researcher.fetch_and_parse_live_content(url))
        self.assertIsNone(content)

    def test_close_client(self): # This tests closing the httpx client
        self._run_async(self.web_researcher.close_client())
        self.mock_http_client.aclose.assert_called_once()


class TestRelevanceScorer(unittest.TestCase):
    def setUp(self):
        self.mock_llm_aggregator = AsyncMock(spec=LLMAggregator)
        # For RelevanceScorer tests, WebResearcher is a dependency passed to the 'score' method.
        # We mock its 'fetch_and_parse_live_content' method.
        self.mock_web_researcher = AsyncMock(spec=WebResearcher)
        self.scorer = RelevanceScorer(llm_aggregator=self.mock_llm_aggregator)
        self.task_description = "Find best practices for Python asyncio."

    def _run_async(self, coro):
        return asyncio.run(coro)

    def test_score_relevance_uses_summary_directly(self):
        # Test case: Tavily summary is good enough (long and high score from Tavily)
        web_search_results = [
            WebSearchResult(url="url1", title="Asyncio Guide", content_summary="Very long and detailed summary about asyncio best practices...", score=0.9)
        ]
        # fetch_and_parse_live_content should NOT be called if summary is used
        self.mock_web_researcher.fetch_and_parse_live_content = AsyncMock()

        mock_llm_score_response = '{"local_relevance_score": 0.95, "local_justification": "Summary was highly relevant."}'
        mock_llm_choice = Choice(index=0, message=OpenHandsMessage(role="assistant", content=mock_llm_score_response), finish_reason="stop")
        mock_llm_response = ChatCompletionResponse(id="score_resp_1", object="chat.completion", created=0, model="sim_model", choices=[mock_llm_choice])
        self.mock_llm_aggregator.chat_completion = AsyncMock(return_value=mock_llm_response)

        processed_results = self._run_async(
            self.scorer.score(web_search_results, self.task_description, self.mock_web_researcher, fetch_full_content_threshold=0.8)
        )

        self.mock_web_researcher.fetch_and_parse_live_content.assert_not_called() # Should use summary
        self.mock_llm_aggregator.chat_completion.assert_called_once()
        self.assertEqual(len(processed_results), 1)
        self.assertEqual(processed_results[0].local_relevance_score, 0.95)
        self.assertEqual(processed_results[0].content_used_for_scoring, web_search_results[0].content_summary)

    def test_score_relevance_fetches_full_content_due_to_low_tavily_score(self):
        web_search_results = [
            WebSearchResult(url="url2", title="Brief Asyncio Note", content_summary="Short but potentially relevant summary.", score=0.5) # score is below threshold
        ]
        mock_full_content = "This is the full, much more detailed content about asyncio that explains everything."
        self.mock_web_researcher.fetch_and_parse_live_content = AsyncMock(return_value=mock_full_content)

        mock_llm_score_response = '{"local_relevance_score": 0.88, "local_justification": "Full content was very relevant."}'
        mock_llm_choice = Choice(index=0, message=OpenHandsMessage(role="assistant", content=mock_llm_score_response), finish_reason="stop")
        mock_llm_response = ChatCompletionResponse(id="score_resp_2", object="chat.completion", created=0, model="sim_model", choices=[mock_llm_choice])
        self.mock_llm_aggregator.chat_completion = AsyncMock(return_value=mock_llm_response)

        processed_results = self._run_async(
            self.scorer.score(web_search_results, self.task_description, self.mock_web_researcher, fetch_full_content_threshold=0.6)
        )

        self.mock_web_researcher.fetch_and_parse_live_content.assert_called_once_with("url2")
        self.mock_llm_aggregator.chat_completion.assert_called_once()
        self.assertEqual(len(processed_results), 1)
        self.assertEqual(processed_results[0].local_relevance_score, 0.88)
        self.assertEqual(processed_results[0].content_used_for_scoring, mock_full_content)

    def test_score_relevance_fetch_full_content_fails_uses_summary(self):
        web_search_results = [WebSearchResult(url="url3", title="Fetch Fail Example", content_summary="Fallback summary.", score=0.4)]
        self.mock_web_researcher.fetch_and_parse_live_content = AsyncMock(return_value=None) # Simulate fetch failure

        mock_llm_score_response = '{"local_relevance_score": 0.6, "local_justification": "Scored based on summary after fetch fail."}'
        mock_llm_choice = Choice(index=0, message=OpenHandsMessage(role="assistant", content=mock_llm_score_response), finish_reason="stop")
        mock_llm_response = ChatCompletionResponse(id="score_resp_3", object="chat.completion", created=0, model="sim_model", choices=[mock_llm_choice])
        self.mock_llm_aggregator.chat_completion = AsyncMock(return_value=mock_llm_response)

        processed_results = self._run_async(
            self.scorer.score(web_search_results, self.task_description, self.mock_web_researcher, fetch_full_content_threshold=0.5)
        )

        self.assertEqual(len(processed_results), 1)
        self.assertEqual(processed_results[0].local_relevance_score, 0.6)
        self.assertEqual(processed_results[0].content_used_for_scoring, "Fallback summary.")
        self.assertIsNone(processed_results[0].error_processing) # Error was in fetching, not local LLM scoring of fallback

    def test_score_relevance_fetch_full_content_fails_no_summary(self):
        web_search_results = [WebSearchResult(url="url4", title="Total Fail Example", content_summary=None, score=0.3)]
        self.mock_web_researcher.fetch_and_parse_live_content = AsyncMock(return_value=None) # Simulate fetch failure

        processed_results = self._run_async(
            self.scorer.score(web_search_results, self.task_description, self.mock_web_researcher, fetch_full_content_threshold=0.5)
        )
        self.assertEqual(len(processed_results), 1)
        self.assertEqual(processed_results[0].local_relevance_score, 0.0)
        self.assertTrue("No content available" in processed_results[0].local_relevance_justification)
        self.assertIsNotNone(processed_results[0].error_processing)
        self.mock_llm_aggregator.chat_completion.assert_not_called()


class TestIntelligentResearchAssistant(unittest.TestCase):
    def setUp(self):
        self.mock_keyword_extractor = AsyncMock(spec=ContextualKeywordExtractor)
        self.mock_web_researcher = AsyncMock(spec=WebResearcher)
        self.mock_relevance_scorer = AsyncMock(spec=RelevanceScorer)
        self.mock_llm_aggregator = AsyncMock(spec=LLMAggregator) # For synthesis

        self.assistant = IntelligentResearchAssistant(
            keyword_extractor=self.mock_keyword_extractor,
            web_researcher=self.mock_web_researcher,
            relevance_scorer=self.mock_relevance_scorer,
            llm_aggregator=self.mock_llm_aggregator
        )

    def _run_async(self, coro):
        return asyncio.run(coro)

    def test_research_for_task_e2e_flow(self):
        research_query = ResearchQuery(original_task_description="Test task: explain Python's GIL.")

        # Mock keyword extraction
        self.mock_keyword_extractor.extract = AsyncMock(return_value=["Python GIL", "threading"])

        # Mock Tavily search
        mock_tavily_report = TavilySearchSessionReport(query_echo="Python GIL", num_results_returned=1, overall_answer="The GIL is a mutex...")
        mock_web_search_results = [WebSearchResult(url="url1", title="GIL Explained", content_summary="Python's GIL...", score=0.9)]
        self.mock_web_researcher.search_with_tavily = AsyncMock(return_value={"report": mock_tavily_report, "results": mock_web_search_results})

        # Mock relevance scoring
        mock_processed_results = [ProcessedResult(source_web_search_result=mock_web_search_results[0], content_used_for_scoring="Python's GIL...", local_relevance_score=0.95, local_relevance_justification="Very relevant")]
        self.mock_relevance_scorer.score = AsyncMock(return_value=mock_processed_results)

        # Mock synthesis (actual LLM call inside synthesize_research)
        mock_llm_synthesis_response = '["The Python Global Interpreter Lock (GIL) is a mutex that protects access to Python objects, preventing multiple native threads from executing Python bytecodes at once."]'
        mock_llm_choice = Choice(index=0, message=OpenHandsMessage(role="assistant", content=mock_llm_synthesis_response), finish_reason="stop")
        mock_llm_response = ChatCompletionResponse(id="synth_resp_1", object="chat.completion", created=0, model="sim_model", choices=[mock_llm_choice])
        self.mock_llm_aggregator.chat_completion = AsyncMock(return_value=mock_llm_response)

        research_output = self._run_async(self.assistant.research_for_task(research_query))

        self.mock_keyword_extractor.extract.assert_called_once()
        self.mock_web_researcher.search_with_tavily.assert_called_once()
        self.mock_relevance_scorer.score.assert_called_once()
        self.mock_llm_aggregator.chat_completion.assert_called_once() # From synthesize_research

        self.assertIsInstance(research_output["session_report"], TavilySearchSessionReport)
        self.assertEqual(len(research_output["knowledge_chunks"]), 1)
        self.assertTrue("Python Global Interpreter Lock" in research_output["knowledge_chunks"][0].content)
        self.assertEqual(research_output["all_web_search_results"], mock_web_search_results)
        self.assertEqual(research_output["processed_results_for_synthesis"], mock_processed_results)

    def test_synthesize_research_uses_correct_content(self):
        # Test that synthesize_research correctly uses content_used_for_scoring
        processed_results = [
            ProcessedResult(
                source_web_search_result=WebSearchResult(url="url1", title="Title 1", content_summary="Summary 1", score=0.8),
                content_used_for_scoring="Full content for URL1 that was scored.", # This should be used
                local_relevance_score=0.9,
                local_relevance_justification="Locally scored as high."
            )
        ]
        research_query = ResearchQuery(original_task_description="Test query")

        # Mock LLM call for synthesis
        self.mock_llm_aggregator.chat_completion = AsyncMock(return_value=ChatCompletionResponse(
            id="synth_resp", choices=[Choice(index=0, message=OpenHandsMessage(role="assistant", content='["Synthesized from full content."]'), finish_reason="stop")]
        ))

        self._run_async(self.assistant.synthesize_research(processed_results, research_query))

        # Check that the prompt sent to LLM for synthesis contains "Full content for URL1"
        # This requires inspecting the arguments to the mocked chat_completion
        call_args = self.mock_llm_aggregator.chat_completion.call_args
        self.assertIsNotNone(call_args)
        request_arg = call_args[0][0] # First positional argument (ChatCompletionRequest)
        prompt_content = request_arg.messages[1].content # User message content
        self.assertIn("Full content for URL1 that was scored.", prompt_content)
        self.assertNotIn("Summary 1", prompt_content)


if __name__ == '__main__':
    unittest.main()

# Example for mocking asyncio.to_thread (if needed, though mocking TavilyClient.search directly is usually simpler):
# @patch('asyncio.to_thread')
# async def test_search_with_tavily_uses_to_thread(self, mock_to_thread):
#     mock_tavily_response = {"results": []} # etc.
#     # Configure mock_to_thread to return a future that resolves to your mock response
#     mock_future = asyncio.Future()
#     mock_future.set_result(mock_tavily_response)
#     mock_to_thread.return_value = mock_future
#
#     await self.web_researcher.search_with_tavily("test query")
#     mock_to_thread.assert_called_once_with(self.mock_tavily_client.search, ...)
        self.mock_web_researcher.fetch_and_parse_live_content = AsyncMock(return_value="Some content")
        self.mock_llm_aggregator.chat_completion = AsyncMock(
            return_value=ChatCompletionResponse(id="score_resp_err", object="chat.completion", created=0, model="sim_model",
                                             choices=[Choice(index=0, message=OpenHandsMessage(role="assistant", content="Not JSON"), finish_reason="stop")])
        )
        processed_results = self._run_async(
            self.scorer.score(search_results, self.task_description, self.mock_web_researcher)
        )
        self.assertEqual(len(processed_results), 1)
        self.assertEqual(processed_results[0].relevance_score, 0.0) # Default on error
        self.assertTrue("LLM scoring failed" in processed_results[0].relevance_justification)


if __name__ == '__main__':
    unittest.main()
