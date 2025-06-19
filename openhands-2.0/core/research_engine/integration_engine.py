import asyncio
import logging
from typing import Dict, Any, Optional, List, TypedDict
from dataclasses import dataclass, field
import re
from datetime import datetime # Added datetime

logger = logging.getLogger(__name__)

@dataclass
class AgentInteraction:
    request_id: str
    user_input: Dict[str, Any]
    agent_actions: List[Dict[str, Any]]
    final_response: Dict[str, Any]
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class KnowledgeNode:
    id: str
    type: str
    content: Any
    source_interaction_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    related_nodes: List[str] = field(default_factory=list)
    confidence: float = 1.0
    scope: str = 'global'
    access_count: int = 0
    last_accessed_ts: Optional[str] = None

class ResearchIntegrationEngine:
    def __init__(self):
        self.name = "ResearchIntegrationEngine"
        self.knowledge_manager = self.KnowledgeManager()
        self.knowledge_retriever = self.KnowledgeRetriever(knowledge_manager_ref=self.knowledge_manager)
        self.arxiv_crawler = self.ArXivCrawler()
        self.github_monitor = self.GitHubMonitor()
        self.paper_classifier = self.PaperClassifier()
        self.auto_implementation = self.AutoImplementation()

    async def initialize(self):
        logger.info(f"Initializing {self.name}...")
        await self.knowledge_manager.initialize()
        await self.knowledge_retriever.initialize()
        await self.arxiv_crawler.initialize()
        await self.github_monitor.initialize()
        await self.paper_classifier.initialize()
        await self.auto_implementation.initialize()
        logger.info(f"{self.name} initialized.")

    async def capture_interaction(self, interaction: AgentInteraction, project_id: Optional[str] = None, session_id: Optional[str] = None) -> Optional[str]:
        return await self.knowledge_manager.capture_interaction(interaction, project_id, session_id)

    async def retrieve_relevant_context(self, query: str, context_type: Optional[str] = None, scope: str = 'global', project_id: Optional[str]=None, session_id: Optional[str]=None) -> List[KnowledgeNode]:
        return await self.knowledge_retriever.retrieve_relevant_context(query, context_type, scope, project_id, session_id)

    class ArXivCrawler:
        async def initialize(self): logger.info("ArXivCrawler initialized.")
        async def get_latest_papers(self, topics: List[str], max_papers: int = 10) -> List[Dict[str, Any]]:
            logger.info(f"Crawling ArXiv for latest papers on {topics} (max {max_papers})...")
            await asyncio.sleep(0.05)
            return [{'id': f'arxiv_{i+1}', 'title': f'Simulated Paper {i+1} on {topics[0] if topics else "AI"}', 'summary': 'Placeholder summary.'} for i in range(min(max_papers, 2))]

    class GitHubMonitor:
        async def initialize(self): logger.info("GitHubMonitor initialized.")
        async def get_trending_repos(self, topics: List[str], max_repos: int = 5) -> List[Dict[str, Any]]:
            logger.info(f"Monitoring GitHub for trending repos on {topics} (max {max_repos})...")
            await asyncio.sleep(0.05)
            return [{'id': f'repo_{i+1}', 'name': f'trending-repo-{i+1}', 'url': 'https://github.com/example/repo', 'description': 'Placeholder description.'} for i in range(min(max_repos, 1))]

    class PaperClassifier:
        async def initialize(self): logger.info("PaperClassifier initialized.")
        async def classify_papers(self, papers: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
            logger.info(f"Classifying {len(papers)} papers...")
            await asyncio.sleep(0.02)
            return {'relevant': papers, 'irrelevant': []}

    class AutoImplementation:
        async def initialize(self): logger.info("AutoImplementation initialized.")
        async def attempt_implementation(self, paper_or_repo: Dict[str, Any]) -> Dict[str, Any]:
            logger.info(f"Attempting auto-implementation of {paper_or_repo.get('title', paper_or_repo.get('name'))}...")
            await asyncio.sleep(0.1)
            return {'status': 'simulated_prototype', 'confidence': 0.65, 'generated_module': 'new_module_placeholder.py'}

    class KnowledgeManager:
        def __init__(self):
            self.global_knowledge: List[KnowledgeNode] = []
            self.project_knowledge: Dict[str, List[KnowledgeNode]] = {}
            self.session_knowledge: Dict[str, List[KnowledgeNode]] = {}
            logger.info("KnowledgeManager component initialized (in-memory with scopes).")

        async def initialize(self):
            await asyncio.sleep(0.01)
            logger.info("KnowledgeManager fully initialized.")

        def _simulate_nlp_annotation(self, text_content: str, num_keywords: int = 5) -> List[str]:
            if not isinstance(text_content, str):
                return []
            words = re.findall(r'\b\w{3,}\b', text_content.lower())
            if not words:
                return []
            from collections import Counter
            common_words = [w for w, _ in Counter(words).most_common(num_keywords)]
            return common_words

        async def capture_interaction(self, interaction: AgentInteraction, project_id: Optional[str] = None, session_id: Optional[str] = None) -> Optional[str]:
            logger.info(f"Capturing interaction {interaction.request_id} into knowledge base...")
            extracted_keywords = []
            response_text_content = None
            if isinstance(interaction.final_response, dict):
                for key in ['content', 'text', 'message', 'summary']:
                    if isinstance(interaction.final_response.get(key), str):
                        response_text_content = interaction.final_response[key]
                        break
                if response_text_content:
                    extracted_keywords = self._simulate_nlp_annotation(response_text_content)
            elif isinstance(interaction.final_response, str):
                response_text_content = interaction.final_response
                extracted_keywords = self._simulate_nlp_annotation(response_text_content)
            node_content = {
                'task_type': interaction.metadata.get('task_type', 'unknown_task'),
                'user_input_summary': str(interaction.user_input)[:150] + ('...' if len(str(interaction.user_input)) > 150 else ''),
                'final_response_summary': str(interaction.final_response)[:150] + ('...' if len(str(interaction.final_response)) > 150 else ''),
                'extracted_keywords_from_response': extracted_keywords,
                'full_interaction_request_id': interaction.request_id
            }
            node_id = f"knode_{interaction.request_id}"
            tags = ['interaction', interaction.metadata.get('task_type', 'unknown_task')] + extracted_keywords
            current_scope = 'global'
            if project_id:
                current_scope = f'project_{project_id}'
            elif session_id:
                current_scope = f'session_{session_id}'
            new_node = KnowledgeNode(
                id=node_id, type='agent_interaction_summary', content=node_content,
                source_interaction_id=interaction.request_id, tags=list(set(tags)), scope=current_scope,
                last_accessed_ts=datetime.utcnow().isoformat()
            )
            if project_id:
                self.project_knowledge.setdefault(project_id, []).append(new_node)
            elif session_id:
                self.session_knowledge.setdefault(session_id, []).append(new_node)
            else:
                self.global_knowledge.append(new_node)
            logger.info(f"Interaction {interaction.request_id} captured as node {node_id} in scope '{current_scope}'.")
            await asyncio.sleep(0.01)
            return node_id

    class KnowledgeRetriever:
        def __init__(self, knowledge_manager_ref: 'ResearchIntegrationEngine.KnowledgeManager'):
            self.knowledge_manager = knowledge_manager_ref
            logger.info("KnowledgeRetriever component initialized.")

        async def initialize(self):
            await asyncio.sleep(0.01)
            logger.info("KnowledgeRetriever fully initialized.")

        async def retrieve_relevant_context(self, query: str, context_type: Optional[str] = None,
                                            scope: str = 'global', project_id: Optional[str] = None,
                                            session_id: Optional[str] = None, min_internal_results: int = 1) -> List[KnowledgeNode]:
            logger.info(f"Retrieving relevant context for query: '{query}' (type: {context_type}, scope: {scope}, project: {project_id}, session: {session_id})...")
            candidate_nodes: List[KnowledgeNode] = []
            # Build candidate list based on scope
            if project_id and project_id in self.knowledge_manager.project_knowledge: # Project specific
                candidate_nodes.extend(self.knowledge_manager.project_knowledge[project_id])
            if session_id and session_id in self.knowledge_manager.session_knowledge: # Session specific
                 candidate_nodes.extend(self.knowledge_manager.session_knowledge[session_id])

            # Include global based on scope parameter or if no specific scope matched
            if scope == 'global' or (scope != 'project' and scope != 'session') or scope == 'all':
                candidate_nodes.extend(self.knowledge_manager.global_knowledge)
            if scope == 'all' and project_id and project_id in self.knowledge_manager.project_knowledge and session_id and session_id in self.knowledge_manager.session_knowledge:
                # Avoid double-adding if already added above due to specific project/session scope with 'all'
                pass # Already covered if 'all' implies adding everything once


            matched_nodes: List[KnowledgeNode] = []
            query_lower = query.lower()
            temp_matched_node_ids = set()

            for node in candidate_nodes:
                if node.id in temp_matched_node_ids:
                    continue
                match_found = False
                if any(query_lower in tag.lower() for tag in node.tags):
                    match_found = True
                if not match_found and isinstance(node.content, dict):
                    for val in node.content.values():
                        if isinstance(val, str) and query_lower in val.lower():
                            match_found = True; break
                        elif isinstance(val, list) and any(isinstance(item, str) and query_lower in item.lower() for item in val):
                            match_found = True; break
                elif not match_found and isinstance(node.content, str) and query_lower in node.content.lower():
                    match_found = True

                if match_found:
                    matched_nodes.append(node)
                    temp_matched_node_ids.add(node.id)

            unique_matched_nodes = [node for i, node in enumerate(matched_nodes) if node.id not in {m.id for m in matched_nodes[:i]}]


            for node in unique_matched_nodes:
                node.access_count += 1
                node.last_accessed_ts = datetime.utcnow().isoformat()

            logger.info(f"Retrieved {len(unique_matched_nodes)} unique internal nodes for query '{query}'.")

            final_results = unique_matched_nodes
            if len(unique_matched_nodes) < min_internal_results:
                logger.info(f"Internal results ({len(unique_matched_nodes)}) less than threshold ({min_internal_results}). Simulating external search...")
                external_search_results_dicts: List[Dict[str, Any]] = []

                # Determine search type (simplified)
                if context_type == 'api_details' or 'api' in query_lower:
                    logger.info("Query suggests API details, using search_api_references.")
                    external_search_results_dicts = await self.search_api_references(query)
                elif 'how to' in query_lower or 'error' in query_lower or 'fix' in query_lower:
                    logger.info("Query suggests troubleshooting, using search_stackoverflow.")
                    external_search_results_dicts = await self.search_stackoverflow(query)
                elif 'code' in query_lower or 'python' in query_lower or 'javascript' in query_lower:
                    logger.info("Query suggests code examples, using search_github_repos.")
                    external_search_results_dicts = await self.search_github_repos(query)
                else:
                    logger.info("Query suggests general documentation, using search_documentation.")
                    external_search_results_dicts = await self.search_documentation(query)

                external_nodes_added_count = 0
                for i, item_dict in enumerate(external_search_results_dicts):
                    node_type_from_item = item_dict.get('type', context_type or 'external_info')
                    # Use a more robust external ID, e.g. from item_dict if it has an 'id' or 'source'
                    ext_node_id_source = item_dict.get('source', f"ext_src_{i}")
                    node_id = f"ext_kn_{node_type_from_item}_{ext_node_id_source}_{query.replace(' ','_')[:20]}"

                    content = item_dict
                    tags = [query, node_type_from_item, item_dict.get('source', 'unknown_external_source')]
                    if isinstance(item_dict.get('title'), str): tags.append(item_dict['title'][:30]) # Add part of title to tags

                    ext_node = KnowledgeNode(id=node_id, type=node_type_from_item, content=content, tags=list(set(tags)), scope='external_sim', confidence=0.7)
                    if ext_node.id not in temp_matched_node_ids:
                        final_results.append(ext_node)
                        temp_matched_node_ids.add(ext_node.id) # Ensure not to add if an internal node had same ID (unlikely)
                        external_nodes_added_count+=1
                logger.info(f"Added {external_nodes_added_count} nodes from simulated external search.")

            return final_results

        async def search_documentation(self, query: str, context: Optional[Dict[str,Any]]=None) -> List[Dict[str,Any]]:
            logger.info(f"(Simulated) Searching documentation for '{query}'...")
            await asyncio.sleep(0.02)
            return [
                {'title': f'API Guide for {query}', 'snippet': f'To use {query}, call Y...', 'source': f'internal_docs_sim/{query.replace(" ","_")}', 'type': 'documentation', 'relevance_score': 0.85},
                {'title': f'Tutorial: Getting started with {query}', 'snippet': f'This tutorial covers basics of {query}...', 'source': f'community_guides_sim/{query.replace(" ","_")}', 'type': 'tutorial'}
            ]

        async def search_stackoverflow(self, query: str, context: Optional[Dict[str,Any]]=None) -> List[Dict[str,Any]]:
            logger.info(f"(Simulated) Searching StackOverflow for '{query}'...")
            await asyncio.sleep(0.02)
            return [
                {'title': f'How to fix {query} issue?', 'answer_snippet': 'Common solution is to check X and Y...', 'votes': 150, 'is_accepted': True, 'source': f'stackoverflow_sim/q12345_{query.replace(" ","_")}', 'type': 'stackoverflow_answer'},
                {'title': f'Understanding {query} behavior', 'answer_snippet': 'The behavior of {query} is due to Z...', 'votes': 75, 'is_accepted': False, 'source': f'stackoverflow_sim/q67890_{query.replace(" ","_")}', 'type': 'stackoverflow_answer'}
            ]

        async def search_github_repos(self, query: str, context: Optional[Dict[str,Any]]=None) -> List[Dict[str,Any]]:
            logger.info(f"(Simulated) Searching GitHub repos for '{query}'...")
            await asyncio.sleep(0.02)
            return [
                {'repo_name': f'{query.replace(" ","_")}_example_project', 'description': f'An example project demonstrating {query}.', 'stars': 200, 'last_commit': '2023-10-01', 'source': f'github_sim/{query.replace(" ","_")}_example', 'type': 'github_repository'},
                {'file_path': f'utils/{query.replace(" ","_")}_helper.py', 'repo_name': f'common_libs_for_{query.replace(" ","_")}', 'snippet': 'def helper_function_for_query(): ...', 'source': f'github_sim/common_libs/{query.replace(" ","_")}_helper.py', 'type': 'github_code_snippet'}
            ]

        async def search_api_references(self, query: str, context: Optional[Dict[str,Any]]=None) -> List[Dict[str,Any]]:
            logger.info(f"(Simulated) Searching API references for '{query}'...")
            await asyncio.sleep(0.02)
            return [
                {'endpoint': f'/api/v1/{query.replace(" ","_")}/get', 'method': 'GET', 'description': f'Retrieves {query} details.', 'request_params': ['id'], 'response_schema': {'id': 'string', 'data': 'object'}, 'source': f'api_docs_sim/{query.replace(" ","_")}', 'type': 'api_reference'}
            ]

__all__ = ['ResearchIntegrationEngine', 'AgentInteraction', 'KnowledgeNode']
