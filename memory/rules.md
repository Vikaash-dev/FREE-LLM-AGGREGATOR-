# AREA-7 Operational Rules

## Core Principles

1. **Scientific Rigor**: All hypotheses must be falsifiable and testable
2. **Minimal Changes**: Implement smallest possible changes to achieve goals
3. **Documentation**: All operations must be logged and documented
4. **Validation**: Test and validate all implementations
5. **Iterative Refinement**: Continuously improve based on results

## Discovery Phase Rules (Sakuna AI Scientist)

### Phase 1: Broad Exploration (Sakuna v1)
- Generate minimum 5 diverse hypotheses per problem
- Each hypothesis must have novelty_score > 0.7
- Run high-throughput experiments in parallel
- Identify anomalous outcomes for deeper investigation

### Phase 2: Deep Investigation (Sakuna v2)
- Focus on single most promising phenomenon
- Design rigorous controlled experiments
- Iterate: Hypothesize -> Experiment -> Analyze -> Refine
- Validate with confidence > 0.8 before implementation

## Implementation Phase Rules (Paper2Code)

### Code Quality Standards
- All code must be modular and well-documented
- Follow PEP 8 for Python code
- Include inline comments linking to source equations/algorithms
- Maintain 80%+ test coverage

### Validation Requirements
- Unit tests for all functions
- Integration tests for pipelines
- Replication tests matching paper results
- Performance benchmarks documented

## Agent Behavior Rules

### Hypothesis Generation
- Use "What if...?" framing
- Score on novelty, feasibility, and impact (0-1 scale)
- Prioritize hypotheses with combined score > 2.1

### Experiment Design
- Define clear control and experimental groups
- Specify measurable metrics
- Set computational budget constraints
- Document expected outcomes with confidence intervals

### Implementation
- Start with language-agnostic pseudocode
- Translate to clean, modular code
- Add comprehensive error handling
- Create replication test suite

## Audit Logging Rules

- Log all operations with structured format
- Include timestamp, agent_id, operation type
- Record input/output data and metrics
- Track success/failure with error messages
- Store logs in audit_logs/ directory

## Memory Management Rules

### Short-Term Memory
- Update after each operation
- Clear completed tasks regularly
- Maintain current session context
- Max size: 1000 entries

### Long-Term Memory
- Archive validated hypotheses
- Store successful patterns
- Document failed patterns (lessons learned)
- Maintain implementation history
- No size limit, but organize by category

## Multi-Agent Coordination Rules

### Agent Personalities (11 Total)
1. **Hypothesis Generator**: Creates novel hypotheses
2. **Experiment Designer**: Designs rigorous experiments
3. **Data Analyzer**: Analyzes experimental results
4. **Code Architect**: Designs system architecture
5. **Implementation Engineer**: Writes production code
6. **Test Engineer**: Creates comprehensive tests
7. **Performance Optimizer**: Optimizes for efficiency
8. **Security Auditor**: Ensures security best practices
9. **Documentation Specialist**: Maintains documentation
10. **Quality Reviewer**: Reviews and validates work
11. **Research Paper Generator**: Generates research papers

### Coordination Protocol
- Agents communicate through structured messages
- Results stored in shared memory
- Conflicts resolved by Quality Reviewer
- All changes require validation

## Continuous Improvement Rules

- Run self-improvement cycle every 100 operations
- Analyze patterns in success/failure logs
- Update rules based on learnings
- Reinvent approach when stuck (after 3 consecutive failures)

## Emergency Protocols

### High Confidence Required (>0.9)
- Production deployments
- Security-related changes
- Data deletion operations

### Rollback Triggers
- Test failure rate > 20%
- Performance degradation > 30%
- Security vulnerabilities detected

### Escalation Protocol
1. Log detailed error information
2. Attempt automatic recovery (max 3 tries)
3. Mark for human review if unresolved
4. Document in failure patterns
