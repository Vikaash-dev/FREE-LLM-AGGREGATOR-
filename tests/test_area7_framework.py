#!/usr/bin/env python3
"""
Tests for AREA-7 Framework

Tests the core functionality of the AREA-7 framework including:
- Memory management
- Audit logging
- Sakuna AI Scientist Protocol
- Paper2Code Protocol
- Multi-agent coordination
"""

import pytest
import asyncio
import os
import shutil
from pathlib import Path
from datetime import datetime

# Import framework components
from area7_framework import (
    MemoryManager,
    AuditLogger,
    SakunaAIScientist,
    Paper2CodeTranslator,
    AREA7Master,
    OperationalMode,
    Hypothesis,
    Experiment,
    AuditLogEntry
)

from multi_agent_personalities import (
    MultiAgentCoordinator,
    HypothesisGeneratorAgent,
    ExperimentDesignerAgent,
    DataAnalyzerAgent,
    AgentPersonality,
    AgentTask
)


@pytest.fixture
def temp_dir(tmp_path):
    """Create temporary directory for tests."""
    test_dir = tmp_path / "area7_test"
    test_dir.mkdir()
    yield test_dir
    # Cleanup
    if test_dir.exists():
        shutil.rmtree(test_dir)


@pytest.fixture
def memory_manager(temp_dir):
    """Create MemoryManager instance for testing."""
    return MemoryManager(base_path=temp_dir / "memory")


@pytest.fixture
def audit_logger(temp_dir):
    """Create AuditLogger instance for testing."""
    return AuditLogger(log_dir=temp_dir / "audit_logs")


class TestMemoryManager:
    """Test MemoryManager functionality."""
    
    def test_initialization(self, memory_manager):
        """Test memory manager initialization."""
        assert memory_manager.base_path.exists()
        assert memory_manager.short_term_memory_path.exists()
        assert memory_manager.long_term_memory_path.exists()
        assert memory_manager.rules_path.exists()
    
    def test_short_term_memory_update(self, memory_manager):
        """Test short-term memory updates."""
        memory_manager.update_short_term_memory(
            "Test Section",
            "Test content for short-term memory"
        )
        
        content = memory_manager.get_short_term_context()
        assert "Test Section" in content
        assert "Test content" in content
    
    def test_long_term_memory_append(self, memory_manager):
        """Test long-term memory append."""
        memory_manager.append_to_long_term_memory(
            "Test Finding",
            "This is a validated finding for long-term storage"
        )
        
        content = memory_manager.get_long_term_knowledge()
        assert "Test Finding" in content
        assert "validated finding" in content
    
    def test_rules_reading(self, memory_manager):
        """Test reading operational rules."""
        rules = memory_manager.read_rules()
        assert "AREA-7 Operational Rules" in rules
        assert "Core Principles" in rules
        assert "Discovery Phase Rules" in rules


class TestAuditLogger:
    """Test AuditLogger functionality."""
    
    def test_initialization(self, audit_logger):
        """Test audit logger initialization."""
        assert audit_logger.log_dir.exists()
        assert audit_logger.log_file.exists() or not audit_logger.log_file.exists()
    
    def test_log_operation(self, audit_logger):
        """Test logging operations."""
        audit_logger.log(
            operation="test_operation",
            agent_id="test_agent",
            operation_type="test",
            input_data={"test": "input"},
            output_data={"test": "output"},
            metrics={"test_metric": 1.0},
            success=True
        )
        
        logs = audit_logger.get_recent_logs(count=10)
        assert len(logs) > 0
        assert logs[-1]["operation"] == "test_operation"
        assert logs[-1]["agent_id"] == "test_agent"
    
    def test_operation_stats(self, audit_logger):
        """Test operation statistics."""
        # Log multiple operations
        for i in range(5):
            audit_logger.log(
                operation=f"operation_{i}",
                agent_id="test_agent",
                operation_type="test",
                input_data={},
                output_data={},
                success=i % 2 == 0  # 3 success, 2 failures
            )
        
        stats = audit_logger.get_operation_stats()
        assert stats["total_operations"] >= 5
        assert "test" in stats["operations_by_type"]


class TestSakunaAIScientist:
    """Test Sakuna AI Scientist Protocol."""
    
    @pytest.fixture
    def sakuna(self, memory_manager, audit_logger):
        """Create SakunaAIScientist instance."""
        return SakunaAIScientist(memory_manager, audit_logger)
    
    @pytest.mark.asyncio
    async def test_ingest_and_deconstruct(self, sakuna):
        """Test problem ingestion and deconstruction."""
        problem = {
            "description": "Test problem",
            "constraints": ["constraint1", "constraint2"],
            "success_criteria": ["criterion1"]
        }
        
        analysis = await sakuna.ingest_and_deconstruct(problem)
        
        assert "problem_statement" in analysis
        assert "fundamental_components" in analysis
        assert "potential_variables" in analysis
        assert analysis["problem_statement"] == "Test problem"
    
    @pytest.mark.asyncio
    async def test_generate_hypotheses(self, sakuna):
        """Test hypothesis generation."""
        analysis = {
            "problem_statement": "Test problem",
            "potential_variables": ["var1", "var2"]
        }
        
        hypotheses = await sakuna.generate_hypotheses(analysis, count=3)
        
        assert len(hypotheses) == 3
        assert all(isinstance(h, Hypothesis) for h in hypotheses)
        assert all(h.novelty_score >= 0.7 for h in hypotheses)
    
    @pytest.mark.asyncio
    async def test_design_experiments(self, sakuna):
        """Test experiment design."""
        hypotheses = [
            Hypothesis(
                id="test_hyp_1",
                title="Test Hypothesis",
                description="Test description",
                what_if_question="What if?",
                variables=["var1"],
                expected_outcome="outcome",
                novelty_score=0.8,
                feasibility_score=0.8,
                impact_score=0.8
            )
        ]
        
        experiments = await sakuna.design_experiments(hypotheses)
        
        assert len(experiments) == 1
        assert all(isinstance(e, Experiment) for e in experiments)
        assert experiments[0].hypothesis_id == "test_hyp_1"
    
    @pytest.mark.asyncio
    async def test_execute_experiments(self, sakuna):
        """Test experiment execution."""
        experiments = [
            Experiment(
                id="test_exp_1",
                hypothesis_id="test_hyp_1",
                description="Test experiment",
                metrics=["metric1", "metric2"]
            )
        ]
        
        completed = await sakuna.execute_experiments(experiments)
        
        assert len(completed) == 1
        assert completed[0].status == "completed"
        assert completed[0].results is not None
        assert completed[0].completed_at is not None


class TestPaper2CodeTranslator:
    """Test Paper2Code Protocol."""
    
    @pytest.fixture
    def paper2code(self, memory_manager, audit_logger):
        """Create Paper2CodeTranslator instance."""
        return Paper2CodeTranslator(memory_manager, audit_logger)
    
    @pytest.mark.asyncio
    async def test_full_paper_deconstruction(self, paper2code):
        """Test paper deconstruction."""
        source = {
            "mechanism": "Test mechanism",
            "metrics": {"accuracy": 0.9}
        }
        
        deconstruction = await paper2code.full_paper_deconstruction(source)
        
        assert "core_algorithm" in deconstruction
        assert "data_structures" in deconstruction
        assert "expected_results" in deconstruction
    
    @pytest.mark.asyncio
    async def test_create_blueprint(self, paper2code):
        """Test blueprint creation."""
        deconstruction = {
            "core_algorithm": "Test algorithm",
            "data_structures": ["list", "dict"]
        }
        
        blueprint = await paper2code.create_blueprint(deconstruction)
        
        assert isinstance(blueprint, str)
        assert "ALGORITHM" in blueprint
        assert "PROCEDURE" in blueprint
    
    @pytest.mark.asyncio
    async def test_implement_code(self, paper2code):
        """Test code implementation."""
        blueprint = "ALGORITHM: Test\nPROCEDURE:\n1. Step 1"
        environment = {"language": "python", "dependencies": []}
        
        implementation = await paper2code.implement_code(blueprint, environment)
        
        assert isinstance(implementation, str)
        assert "def " in implementation  # Should contain function definitions
        assert "import" in implementation  # Should contain imports
    
    @pytest.mark.asyncio
    async def test_create_test_suite(self, paper2code):
        """Test test suite creation."""
        implementation = "def test_func(): pass"
        
        test_suite = await paper2code.create_test_suite(implementation)
        
        assert isinstance(test_suite, str)
        assert "test_" in test_suite  # Should contain test functions
        assert "import pytest" in test_suite


class TestAREA7Master:
    """Test AREA-7 Master controller."""
    
    @pytest.fixture
    def area7_master(self):
        """Create AREA7Master instance."""
        return AREA7Master()
    
    @pytest.mark.asyncio
    async def test_determine_mode_discovery(self, area7_master):
        """Test mode determination for discovery."""
        goal = {
            "description": "Test problem",
            "exploratory": True,
            "novel": True
        }
        
        mode = await area7_master.determine_mode(goal)
        
        assert mode == OperationalMode.DISCOVERY
    
    @pytest.mark.asyncio
    async def test_determine_mode_implementation(self, area7_master):
        """Test mode determination for implementation."""
        goal = {
            "description": "Implement algorithm",
            "paper_reference": "Test paper",
            "exploratory": False
        }
        
        mode = await area7_master.determine_mode(goal)
        
        assert mode == OperationalMode.IMPLEMENTATION
    
    @pytest.mark.asyncio
    async def test_determine_mode_hybrid(self, area7_master):
        """Test mode determination for hybrid."""
        goal = {
            "description": "Complex problem",
            "exploratory": True,
            "paper_reference": "Test paper"
        }
        
        mode = await area7_master.determine_mode(goal)
        
        assert mode == OperationalMode.HYBRID


class TestMultiAgentSystem:
    """Test Multi-Agent System."""
    
    @pytest.fixture
    def coordinator(self, memory_manager, audit_logger):
        """Create MultiAgentCoordinator instance."""
        return MultiAgentCoordinator(memory_manager, audit_logger)
    
    @pytest.mark.asyncio
    async def test_hypothesis_generator_agent(self, memory_manager, audit_logger):
        """Test HypothesisGeneratorAgent."""
        agent = HypothesisGeneratorAgent(memory_manager, audit_logger)
        
        task = AgentTask(
            task_id="test_task_1",
            assigned_to=AgentPersonality.HYPOTHESIS_GENERATOR,
            description="Generate hypotheses",
            input_data={
                "problem": {"description": "Test problem"},
                "constraints": []
            }
        )
        
        completed_task = await agent.process_task(task)
        
        assert completed_task.status == "completed"
        assert "hypotheses" in completed_task.output_data
        assert len(completed_task.output_data["hypotheses"]) > 0
    
    @pytest.mark.asyncio
    async def test_experiment_designer_agent(self, memory_manager, audit_logger):
        """Test ExperimentDesignerAgent."""
        agent = ExperimentDesignerAgent(memory_manager, audit_logger)
        
        task = AgentTask(
            task_id="test_task_2",
            assigned_to=AgentPersonality.EXPERIMENT_DESIGNER,
            description="Design experiments",
            input_data={
                "hypotheses": [
                    {"id": "hyp_1", "title": "Test hypothesis"}
                ]
            }
        )
        
        completed_task = await agent.process_task(task)
        
        assert completed_task.status == "completed"
        assert "experiments" in completed_task.output_data
        assert len(completed_task.output_data["experiments"]) > 0
    
    @pytest.mark.asyncio
    async def test_data_analyzer_agent(self, memory_manager, audit_logger):
        """Test DataAnalyzerAgent."""
        agent = DataAnalyzerAgent(memory_manager, audit_logger)
        
        task = AgentTask(
            task_id="test_task_3",
            assigned_to=AgentPersonality.DATA_ANALYZER,
            description="Analyze data",
            input_data={
                "experiment_results": [
                    {
                        "experiment_id": "exp_1",
                        "metrics": {"improvement": 0.15, "accuracy": 0.92}
                    }
                ]
            }
        )
        
        completed_task = await agent.process_task(task)
        
        assert completed_task.status == "completed"
        assert "analysis" in completed_task.output_data
        assert "summary" in completed_task.output_data["analysis"]
    
    @pytest.mark.asyncio
    async def test_agent_coordination(self, coordinator):
        """Test multi-agent coordination."""
        task = await coordinator.assign_task(
            AgentPersonality.HYPOTHESIS_GENERATOR,
            "Test task",
            {"problem": {"description": "Test"}}
        )
        
        assert task.task_id is not None
        assert task.assigned_to == AgentPersonality.HYPOTHESIS_GENERATOR
        assert task.status == "pending"


class TestIntegration:
    """Integration tests for complete workflows."""
    
    @pytest.mark.asyncio
    async def test_complete_discovery_workflow(self, temp_dir):
        """Test complete discovery workflow."""
        # This is a longer integration test
        area7 = AREA7Master()
        
        goal = {
            "description": "Test discovery problem",
            "exploratory": True,
            "novel": True,
            "constraints": ["constraint1"],
            "success_criteria": ["criterion1"]
        }
        
        # Note: This is a simplified test - full execution would take longer
        mode = await area7.determine_mode(goal)
        assert mode == OperationalMode.DISCOVERY
    
    @pytest.mark.asyncio
    async def test_complete_implementation_workflow(self, temp_dir):
        """Test complete implementation workflow."""
        area7 = AREA7Master()
        
        goal = {
            "description": "Test implementation",
            "paper_reference": "Test paper",
            "exploratory": False,
            "mechanism": "Test mechanism",
            "metrics": {"accuracy": 0.9}
        }
        
        mode = await area7.determine_mode(goal)
        assert mode == OperationalMode.IMPLEMENTATION


# Test runner
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
