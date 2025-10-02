"""
ADA-7 Framework Core Orchestrator

Main orchestrator class that coordinates all 7 stages of the development process.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ProjectContext:
    """Context information for a project being developed."""
    
    project_id: str
    name: str
    description: str
    constraints: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # Stage results
    stage1_results: Optional[Dict[str, Any]] = None
    stage2_results: Optional[Dict[str, Any]] = None
    stage3_results: Optional[Dict[str, Any]] = None
    stage4_results: Optional[Dict[str, Any]] = None
    stage5_results: Optional[Dict[str, Any]] = None
    stage6_results: Optional[Dict[str, Any]] = None
    stage7_results: Optional[Dict[str, Any]] = None
    
    # Current stage
    current_stage: int = 0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AcademicReference:
    """Academic paper reference."""
    
    authors: str
    year: int
    title: str
    arxiv_id: str
    relevance_score: float = 0.0
    citation_count: int = 0
    
    def format_citation(self) -> str:
        """Format as citation string."""
        return f"[{self.authors}, {self.year}, \"{self.title}\", arXiv:{self.arxiv_id}]"


@dataclass
class GitHubRepository:
    """GitHub repository reference."""
    
    owner: str
    name: str
    url: str
    stars: int
    last_commit_days_ago: int
    description: str = ""
    
    def format_reference(self) -> str:
        """Format as reference string."""
        return f"[{self.owner}/{self.name}]({self.url}) - {self.stars}★ - Updated {self.last_commit_days_ago} days ago"


@dataclass
class DecisionMatrix:
    """Decision matrix for comparing alternatives."""
    
    alternatives: List[str]
    criteria: List[str]
    scores: Dict[str, Dict[str, float]]  # alternative -> criterion -> score
    weights: Dict[str, float]  # criterion -> weight
    
    def calculate_weighted_scores(self) -> Dict[str, float]:
        """Calculate weighted scores for each alternative."""
        weighted_scores = {}
        
        for alternative in self.alternatives:
            score = 0.0
            for criterion, weight in self.weights.items():
                if criterion in self.scores.get(alternative, {}):
                    score += self.scores[alternative][criterion] * weight
            weighted_scores[alternative] = score
        
        return weighted_scores
    
    def get_best_alternative(self) -> str:
        """Get the alternative with the highest weighted score."""
        weighted_scores = self.calculate_weighted_scores()
        return max(weighted_scores.items(), key=lambda x: x[1])[0]


class ADA7Framework:
    """
    Advanced Development Assistant (ADA-7) Framework
    
    Orchestrates the 7-stage evolutionary development methodology with
    academic research integration and evidence-based decision making.
    """
    
    def __init__(self, aggregator=None, meta_controller=None):
        """
        Initialize ADA-7 framework.
        
        Args:
            aggregator: LLM aggregator for accessing various models
            meta_controller: Meta-controller for intelligent model selection
        """
        self.aggregator = aggregator
        self.meta_controller = meta_controller
        
        # Initialize stage handlers (lazy loading)
        self._stage_handlers = {}
        
        # Active projects
        self.projects: Dict[str, ProjectContext] = {}
        
        # Knowledge base
        self.academic_papers: List[AcademicReference] = []
        self.github_repositories: List[GitHubRepository] = []
        
        # Configuration
        self.config = {
            "min_arxiv_papers_per_decision": 2,
            "min_github_repos_per_decision": 3,
            "competitive_analysis_count": 10,
            "architecture_alternatives": 3,
            "test_coverage_target": 0.80,
        }
        
        logger.info("ADA-7 Framework initialized")
    
    def _get_stage_handler(self, stage_number: int):
        """Get or create stage handler."""
        if stage_number not in self._stage_handlers:
            # Lazy import to avoid circular dependencies
            from .stage1_requirements import Stage1RequirementsAnalysis
            from .stage2_architecture import Stage2ArchitectureDesign
            from .stage3_components import Stage3ComponentDesign
            from .stage4_implementation import Stage4ImplementationStrategy
            from .stage5_testing import Stage5TestingFramework
            from .stage6_deployment import Stage6DeploymentManagement
            from .stage7_maintenance import Stage7Maintenance
            
            handlers = {
                1: Stage1RequirementsAnalysis,
                2: Stage2ArchitectureDesign,
                3: Stage3ComponentDesign,
                4: Stage4ImplementationStrategy,
                5: Stage5TestingFramework,
                6: Stage6DeploymentManagement,
                7: Stage7Maintenance,
            }
            
            if stage_number in handlers:
                self._stage_handlers[stage_number] = handlers[stage_number](
                    framework=self,
                    aggregator=self.aggregator,
                    meta_controller=self.meta_controller
                )
        
        return self._stage_handlers.get(stage_number)
    
    async def start_project(
        self,
        name: str,
        description: str,
        constraints: Optional[Dict[str, Any]] = None
    ) -> ProjectContext:
        """
        Start a new project development cycle.
        
        Args:
            name: Project name
            description: Project description
            constraints: Project constraints (budget, timeline, team_size, etc.)
        
        Returns:
            ProjectContext: Created project context
        """
        project_id = f"proj_{datetime.utcnow().timestamp()}"
        
        project = ProjectContext(
            project_id=project_id,
            name=name,
            description=description,
            constraints=constraints or {}
        )
        
        self.projects[project_id] = project
        
        logger.info(
            f"Started new project: {name}",
            extra={"project_id": project_id}
        )
        
        return project
    
    async def execute_stage_1(
        self,
        project: ProjectContext
    ) -> Dict[str, Any]:
        """
        Execute Stage 1: Requirements Analysis & Competitive Intelligence.
        
        Args:
            project: Project context
        
        Returns:
            Stage 1 results including user stories, competitive analysis, and requirements
        """
        logger.info(
            f"Executing Stage 1 for project: {project.name}",
            extra={"project_id": project.project_id}
        )
        
        stage_handler = self._get_stage_handler(1)
        results = await stage_handler.execute(project)
        
        project.stage1_results = results
        project.current_stage = 1
        
        return results
    
    async def execute_stage_2(
        self,
        project: ProjectContext,
        stage1_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute Stage 2: Architecture Design & Academic Validation.
        
        Args:
            project: Project context
            stage1_results: Results from Stage 1 (optional, uses project context if not provided)
        
        Returns:
            Stage 2 results including architecture variants and validation
        """
        if stage1_results:
            project.stage1_results = stage1_results
        
        if not project.stage1_results:
            raise ValueError("Stage 1 must be completed before Stage 2")
        
        logger.info(
            f"Executing Stage 2 for project: {project.name}",
            extra={"project_id": project.project_id}
        )
        
        stage_handler = self._get_stage_handler(2)
        results = await stage_handler.execute(project)
        
        project.stage2_results = results
        project.current_stage = 2
        
        return results
    
    async def execute_stage_3(
        self,
        project: ProjectContext,
        stage2_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute Stage 3: Component Design & Technology Stack.
        
        Args:
            project: Project context
            stage2_results: Results from Stage 2 (optional)
        
        Returns:
            Stage 3 results including component breakdown and technology selection
        """
        if stage2_results:
            project.stage2_results = stage2_results
        
        if not project.stage2_results:
            raise ValueError("Stage 2 must be completed before Stage 3")
        
        logger.info(
            f"Executing Stage 3 for project: {project.name}",
            extra={"project_id": project.project_id}
        )
        
        stage_handler = self._get_stage_handler(3)
        results = await stage_handler.execute(project)
        
        project.stage3_results = results
        project.current_stage = 3
        
        return results
    
    async def execute_stage_4(
        self,
        project: ProjectContext,
        stage3_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute Stage 4: Implementation Strategy & Development Pipeline.
        
        Args:
            project: Project context
            stage3_results: Results from Stage 3 (optional)
        
        Returns:
            Stage 4 results including phased development plan and CI/CD setup
        """
        if stage3_results:
            project.stage3_results = stage3_results
        
        if not project.stage3_results:
            raise ValueError("Stage 3 must be completed before Stage 4")
        
        logger.info(
            f"Executing Stage 4 for project: {project.name}",
            extra={"project_id": project.project_id}
        )
        
        stage_handler = self._get_stage_handler(4)
        results = await stage_handler.execute(project)
        
        project.stage4_results = results
        project.current_stage = 4
        
        return results
    
    async def execute_stage_5(
        self,
        project: ProjectContext,
        stage4_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute Stage 5: Testing Framework & Quality Assurance.
        
        Args:
            project: Project context
            stage4_results: Results from Stage 4 (optional)
        
        Returns:
            Stage 5 results including testing strategy and quality gates
        """
        if stage4_results:
            project.stage4_results = stage4_results
        
        if not project.stage4_results:
            raise ValueError("Stage 4 must be completed before Stage 5")
        
        logger.info(
            f"Executing Stage 5 for project: {project.name}",
            extra={"project_id": project.project_id}
        )
        
        stage_handler = self._get_stage_handler(5)
        results = await stage_handler.execute(project)
        
        project.stage5_results = results
        project.current_stage = 5
        
        return results
    
    async def execute_stage_6(
        self,
        project: ProjectContext,
        stage5_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute Stage 6: Deployment & Infrastructure Management.
        
        Args:
            project: Project context
            stage5_results: Results from Stage 5 (optional)
        
        Returns:
            Stage 6 results including deployment strategy and infrastructure setup
        """
        if stage5_results:
            project.stage5_results = stage5_results
        
        if not project.stage5_results:
            raise ValueError("Stage 5 must be completed before Stage 6")
        
        logger.info(
            f"Executing Stage 6 for project: {project.name}",
            extra={"project_id": project.project_id}
        )
        
        stage_handler = self._get_stage_handler(6)
        results = await stage_handler.execute(project)
        
        project.stage6_results = results
        project.current_stage = 6
        
        return results
    
    async def execute_stage_7(
        self,
        project: ProjectContext,
        stage6_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute Stage 7: Maintenance & Continuous Evolution.
        
        Args:
            project: Project context
            stage6_results: Results from Stage 6 (optional)
        
        Returns:
            Stage 7 results including maintenance strategy and evolution roadmap
        """
        if stage6_results:
            project.stage6_results = stage6_results
        
        if not project.stage6_results:
            raise ValueError("Stage 6 must be completed before Stage 7")
        
        logger.info(
            f"Executing Stage 7 for project: {project.name}",
            extra={"project_id": project.project_id}
        )
        
        stage_handler = self._get_stage_handler(7)
        results = await stage_handler.execute(project)
        
        project.stage7_results = results
        project.current_stage = 7
        
        return results
    
    async def execute_all_stages(
        self,
        project: ProjectContext
    ) -> Dict[str, Any]:
        """
        Execute all 7 stages in sequence.
        
        Args:
            project: Project context
        
        Returns:
            Combined results from all stages
        """
        logger.info(
            f"Executing all stages for project: {project.name}",
            extra={"project_id": project.project_id}
        )
        
        results = {
            "project_id": project.project_id,
            "project_name": project.name,
            "stages": {}
        }
        
        try:
            # Execute stages sequentially
            results["stages"]["stage1"] = await self.execute_stage_1(project)
            results["stages"]["stage2"] = await self.execute_stage_2(project)
            results["stages"]["stage3"] = await self.execute_stage_3(project)
            results["stages"]["stage4"] = await self.execute_stage_4(project)
            results["stages"]["stage5"] = await self.execute_stage_5(project)
            results["stages"]["stage6"] = await self.execute_stage_6(project)
            results["stages"]["stage7"] = await self.execute_stage_7(project)
            
            results["status"] = "completed"
            results["completed_at"] = datetime.utcnow().isoformat()
            
            logger.info(
                f"Completed all stages for project: {project.name}",
                extra={"project_id": project.project_id}
            )
            
        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)
            logger.error(
                f"Failed to complete stages for project: {project.name}",
                extra={"project_id": project.project_id, "error": str(e)},
                exc_info=True
            )
        
        return results
    
    def add_academic_reference(self, reference: AcademicReference):
        """Add an academic paper reference to the knowledge base."""
        self.academic_papers.append(reference)
        logger.debug(f"Added academic reference: {reference.format_citation()}")
    
    def add_github_repository(self, repository: GitHubRepository):
        """Add a GitHub repository reference to the knowledge base."""
        self.github_repositories.append(repository)
        logger.debug(f"Added GitHub repository: {repository.format_reference()}")
    
    def search_academic_papers(
        self,
        keywords: List[str],
        min_relevance: float = 0.5
    ) -> List[AcademicReference]:
        """
        Search academic papers by keywords.
        
        Args:
            keywords: List of keywords to search for
            min_relevance: Minimum relevance score
        
        Returns:
            List of matching academic references
        """
        # Simple keyword matching (can be enhanced with semantic search)
        matching_papers = []
        
        for paper in self.academic_papers:
            relevance = 0.0
            searchable_text = f"{paper.title} {paper.authors}".lower()
            
            for keyword in keywords:
                if keyword.lower() in searchable_text:
                    relevance += 0.2
            
            if relevance >= min_relevance:
                paper.relevance_score = relevance
                matching_papers.append(paper)
        
        # Sort by relevance and citation count
        matching_papers.sort(
            key=lambda p: (p.relevance_score, p.citation_count),
            reverse=True
        )
        
        return matching_papers
    
    def search_github_repositories(
        self,
        keywords: List[str],
        min_stars: int = 1000
    ) -> List[GitHubRepository]:
        """
        Search GitHub repositories by keywords.
        
        Args:
            keywords: List of keywords to search for
            min_stars: Minimum star count
        
        Returns:
            List of matching GitHub repositories
        """
        matching_repos = []
        
        for repo in self.github_repositories:
            if repo.stars < min_stars:
                continue
            
            searchable_text = f"{repo.name} {repo.description}".lower()
            
            for keyword in keywords:
                if keyword.lower() in searchable_text:
                    matching_repos.append(repo)
                    break
        
        # Sort by stars
        matching_repos.sort(key=lambda r: r.stars, reverse=True)
        
        return matching_repos
    
    def create_decision_matrix(
        self,
        alternatives: List[str],
        criteria: List[str],
        weights: Dict[str, float]
    ) -> DecisionMatrix:
        """
        Create a decision matrix for comparing alternatives.
        
        Args:
            alternatives: List of alternatives to compare
            criteria: List of criteria to evaluate
            weights: Weights for each criterion (must sum to 1.0)
        
        Returns:
            DecisionMatrix instance
        """
        # Validate weights
        weight_sum = sum(weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            logger.warning(
                f"Criterion weights sum to {weight_sum}, normalizing to 1.0"
            )
            weights = {k: v / weight_sum for k, v in weights.items()}
        
        return DecisionMatrix(
            alternatives=alternatives,
            criteria=criteria,
            scores={},
            weights=weights
        )
    
    def get_project_status(self, project_id: str) -> Dict[str, Any]:
        """
        Get status of a project.
        
        Args:
            project_id: Project ID
        
        Returns:
            Project status information
        """
        project = self.projects.get(project_id)
        
        if not project:
            return {"error": "Project not found"}
        
        return {
            "project_id": project.project_id,
            "name": project.name,
            "description": project.description,
            "current_stage": project.current_stage,
            "created_at": project.created_at.isoformat(),
            "stages_completed": [
                i for i in range(1, 8)
                if getattr(project, f"stage{i}_results") is not None
            ],
            "constraints": project.constraints,
        }
    
    def list_projects(self) -> List[Dict[str, Any]]:
        """
        List all projects.
        
        Returns:
            List of project status information
        """
        return [
            self.get_project_status(project_id)
            for project_id in self.projects.keys()
        ]
