# Rules for Component Evaluation and Documentation

## Core Principles

### 1. Evaluation Standards
- Every component MUST be evaluated against 6 similar GitHub projects (with >500 stars)
- Every component MUST be cross-referenced with 8 relevant arXiv papers (2019-2025)
- All evaluations must include quantitative metrics where applicable
- Citations must follow format: `[Author et al., Year, "Title", arXiv:ID]` for papers
- Citations must include: `[Project Name, Stars, Last Update, GitHub URL]` for repositories

### 2. Documentation Requirements
- All component analyses must be stored in markdown files
- Short-term memory (short_term_memory.md) tracks current evaluation progress
- Long-term memory (long_term_memory.md) stores persistent findings and decisions
- Each component evaluation must include:
  - Component overview and responsibilities
  - Architecture analysis
  - Implementation patterns identified
  - Performance characteristics
  - Security considerations
  - Comparison with research and projects
  - Recommendations for improvement

### 3. Research Citation Standards
- Academic papers must be from reputable sources (arXiv, ACL, NeurIPS, ICML, etc.)
- Papers should be relevant to specific component functionality
- GitHub projects must be actively maintained (commits within last 6 months preferred)
- Projects must have clear documentation and production usage evidence

### 4. Cross-Reference Methodology
For each component:
1. Identify core functionality and design patterns
2. Search for 6 similar GitHub implementations
3. Analyze implementation differences and trade-offs
4. Search for 8 relevant research papers
5. Extract key insights from papers
6. Synthesize findings into actionable recommendations
7. Document in component-specific evaluation file

### 5. Component Categories
Components are evaluated in these categories:
- **Core Orchestration**: Aggregator, Router, MetaController
- **Resource Management**: AccountManager, RateLimiter, StateTracker
- **Advanced Features**: EnsembleSystem, AutoUpdater, Researcher
- **Provider Implementations**: BaseProvider and specific providers
- **API Layer**: Server, endpoints, middleware
- **Utilities**: Browser monitoring, planning, code generation

### 6. Evaluation Metrics
Each component evaluation must include:
- **Performance**: Latency, throughput, resource usage
- **Reliability**: Error rates, availability, fault tolerance
- **Scalability**: Horizontal/vertical scaling capabilities
- **Maintainability**: Code complexity, documentation quality
- **Security**: Vulnerability assessment, best practices adherence
- **Innovation**: Novel approaches compared to existing solutions

### 7. Memory File Usage
- **short_term_memory.md**: 
  - Current component being evaluated
  - In-progress research findings
  - Temporary notes and observations
  - Next steps and pending tasks
  
- **long_term_memory.md**:
  - Completed component evaluations summary
  - Key architectural decisions
  - Persistent patterns and anti-patterns found
  - Cross-component insights
  - Overall project recommendations

### 8. Update Frequency
- Short-term memory: Updated after each component evaluation
- Long-term memory: Updated when component evaluation is complete
- Rules: Updated only when methodology changes are agreed upon

### 9. Quality Gates
Before marking a component evaluation as complete:
- [ ] All 6 GitHub projects identified and analyzed
- [ ] All 8 research papers reviewed and cited
- [ ] Implementation comparison table created
- [ ] Recommendations documented with rationale
- [ ] Findings added to long-term memory
- [ ] Next component identified

### 10. Collaboration Standards
- All findings are evidence-based with citations
- Recommendations include confidence levels (High/Medium/Low)
- Trade-offs are explicitly documented
- Alternative approaches are considered
- Implementation complexity is assessed

## Enforcement
These rules MUST be followed for all component evaluations. Any deviation must be documented with rationale in the evaluation file.
