"""
ADA-7 Advanced Development Assistant Framework

This module implements the 7-stage evolutionary development methodology
for multi-project software development with academic research integration.
"""

from .framework import ADA7Framework
from .stage1_requirements import Stage1RequirementsAnalysis
from .stage2_architecture import Stage2ArchitectureDesign
from .stage3_components import Stage3ComponentDesign
from .stage4_implementation import Stage4ImplementationStrategy
from .stage5_testing import Stage5TestingFramework
from .stage6_deployment import Stage6DeploymentManagement
from .stage7_maintenance import Stage7Maintenance

__all__ = [
    'ADA7Framework',
    'Stage1RequirementsAnalysis',
    'Stage2ArchitectureDesign',
    'Stage3ComponentDesign',
    'Stage4ImplementationStrategy',
    'Stage5TestingFramework',
    'Stage6DeploymentManagement',
    'Stage7Maintenance',
]
