import unittest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import httpx # For httpx.Response and potential errors

from src.core.research_components import ContextualKeywordExtractor, WebResearcher, RelevanceScorer
from src.core.research_structures import SearchResult, ProcessedResult
from src.models import ChatCompletionResponse, Choice, Message as OpenHandsMessage
# Assuming LLMAggregator is needed for ContextualKeywordExtractor and RelevanceScorer
from src.core.aggregator import LLMAggregator


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
        self.mock_http_client = AsyncMock(spec=httpx.AsyncClient)
        self.web_researcher = WebResearcher(http_client=self.mock_http_client)

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

    def test_search_success(self):
        keywords = ["python asyncio"]
        mock_html_content = """
        <html><body>
            <a href="/url?q=https://example.com/async_page1&sa=U"><h3>Async Python Guide</h3></a>
            <div><span>Snippet for async page 1...</span></div>
            <a href="/url?q=https://example.com/async_page2&sa=U"><h3>Another Async Tip</h3></a>
            <div><span>Snippet for async page 2...</span></div>
        </body></html>
        """
        self.mock_http_client.get = AsyncMock(return_value=self._prepare_mock_response(mock_html_content))

        results = self._run_async(self.web_researcher.search(keywords))

        self.mock_http_client.get.assert_called_once_with(
            f"https://www.google.com/search?q={httpx.utils.quote(keywords[0])}",
            headers=self.web_researcher.headers
        )
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], SearchResult)
        self.assertEqual(results[0].source_identifier, "https://example.com/async_page1")
        self.assertEqual(results[0].title, "Async Python Guide")
        self.assertTrue("Snippet for async page 1" in results[0].snippet if results[0].snippet else False)

    def test_search_http_error(self):
        keywords = ["test keyword"]
        self.mock_http_client.get = AsyncMock(return_value=self._prepare_mock_response("Error page", status_code=500))

        results = self._run_async(self.web_researcher.search(keywords))
        self.assertEqual(len(results), 0)

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

    def test_close_client(self):
        self._run_async(self.web_researcher.close_client())
        self.mock_http_client.aclose.assert_called_once()


class TestRelevanceScorer(unittest.TestCase):
    def setUp(self):
        self.mock_llm_aggregator = AsyncMock(spec=LLMAggregator)
        # WebResearcher needs an httpx client, even if its methods are mocked.
        # We can pass a MagicMock for the client if WebResearcher itself is fully mocked.
        self.mock_web_researcher = AsyncMock(spec=WebResearcher)
        self.scorer = RelevanceScorer(llm_aggregator=self.mock_llm_aggregator)
        self.task_description = "Find best practices for Python asyncio."

    def _run_async(self, coro):
        return asyncio.run(coro)

    def test_score_relevance_success(self):
        search_results = [
            SearchResult(source_identifier="url1", title="Asyncio Basics", snippet="...")
        ]
        mock_fetched_content = "Detailed content about asyncio best practices..."
        self.mock_web_researcher.fetch_and_parse_live_content = AsyncMock(return_value=mock_fetched_content)

        mock_llm_score_response = '{"relevance_score": 0.9, "justification": "Highly relevant."}'
        mock_llm_choice = Choice(index=0, message=OpenHandsMessage(role="assistant", content=mock_llm_score_response), finish_reason="stop")
        mock_llm_response = ChatCompletionResponse(id="score_resp_1", object="chat.completion", created=0, model="sim_model", choices=[mock_llm_choice])
        self.mock_llm_aggregator.chat_completion = AsyncMock(return_value=mock_llm_response)

        processed_results = self._run_async(
            self.scorer.score(search_results, self.task_description, self.mock_web_researcher)
        )

        self.mock_web_researcher.fetch_and_parse_live_content.assert_called_once_with("url1")
        self.mock_llm_aggregator.chat_completion.assert_called_once()
        self.assertEqual(len(processed_results), 1)
        self.assertIsInstance(processed_results[0], ProcessedResult)
        self.assertEqual(processed_results[0].relevance_score, 0.9)
        self.assertEqual(processed_results[0].relevance_justification, "Highly relevant.")
        self.assertEqual(processed_results[0].fetched_content, mock_fetched_content)

    def test_score_relevance_fetch_failed(self):
        search_results = [SearchResult(source_identifier="url_fetch_fail")]
        self.mock_web_researcher.fetch_and_parse_live_content = AsyncMock(return_value=None) # Simulate fetch failure

        processed_results = self._run_async(
            self.scorer.score(search_results, self.task_description, self.mock_web_researcher)
        )

        self.assertEqual(len(processed_results), 1)
        self.assertEqual(processed_results[0].relevance_score, 0.0)
        self.assertTrue("Failed to fetch content" in processed_results[0].relevance_justification)
        self.assertIsNotNone(processed_results[0].error_fetching)
        self.mock_llm_aggregator.chat_completion.assert_not_called() # LLM should not be called if content fetch fails

    def test_score_relevance_llm_json_error(self):
        search_results = [SearchResult(source_identifier="url_json_err")]
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
