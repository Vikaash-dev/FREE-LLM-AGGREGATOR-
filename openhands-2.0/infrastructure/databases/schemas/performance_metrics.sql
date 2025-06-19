-- Placeholder for performance_metrics table schema
-- This schema will be expanded based on the actual metrics collected
-- by the PerformanceOptimizer and other components.

CREATE TABLE IF NOT EXISTS performance_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    task_id VARCHAR(255) UNIQUE,
    agent_name VARCHAR(100),
    execution_strategy VARCHAR(50),
    duration_seconds FLOAT,
    success BOOLEAN,
    complexity_score FLOAT,
    resource_memory_mb INTEGER,
    resource_cpu_cores FLOAT,
    error_message TEXT,
    metadata JSONB -- For any additional structured data
);

CREATE INDEX IF NOT EXISTS idx_perf_metrics_timestamp ON performance_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_perf_metrics_task_id ON performance_metrics(task_id);
CREATE INDEX IF NOT EXISTS idx_perf_metrics_agent_name ON performance_metrics(agent_name);

-- Example: Other potential tables related to system monitoring
-- CREATE TABLE IF NOT EXISTS system_load (
--     id SERIAL PRIMARY KEY,
--     timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
--     cpu_utilization_percent FLOAT,
--     memory_utilization_percent FLOAT,
--     active_tasks INTEGER
-- );
