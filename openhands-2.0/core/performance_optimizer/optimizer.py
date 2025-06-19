import asyncio
import logging
from typing import Dict, Any, Optional, Callable, Tuple # Added Tuple
from contextlib import contextmanager, asynccontextmanager
import hashlib # For creating more robust cache keys from dicts
import json # For serializing dicts for hashing
from datetime import datetime # For cache timestamp
import uuid # For error key generation

logger = logging.getLogger(__name__)

class PerformanceOptimizer:
    def __init__(self):
        self.name = "PerformanceOptimizer"
        self.request_stats: Dict[str, Dict[str, Any]] = {}
        self.cache_stats: Dict[str, int] = {'hits': 0, 'misses': 0, 'errors': 0}
        self.mock_cache: Dict[Tuple[Any, ...], Any] = {} # Cache key will be a tuple
        # self.load_balancer = LoadBalancer()
        # self.profiler = Profiler()

    async def initialize(self):
        logger.info(f"Initializing {self.name}...")
        await asyncio.sleep(0.01)
        logger.info(f"{self.name} initialized.")

    @asynccontextmanager
    async def monitor_execution(self, task_name: Optional[str] = None):
        start_time = asyncio.get_event_loop().time()
        try:
            yield
        finally:
            end_time = asyncio.get_event_loop().time()
            duration = end_time - start_time
            if task_name:
                logger.info(f"Task '{task_name}' executed in {duration:.4f} seconds.")
                self.record_task_performance(task_name, duration)
            else:
                logger.info(f"Execution block completed in {duration:.4f} seconds.")

    @contextmanager
    def monitor_execution_sync(self, task_name: Optional[str] = None):
        import time
        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration = end_time - start_time
            if task_name:
                logger.info(f"Task '{task_name}' (sync) executed in {duration:.4f} seconds.")
                self.record_task_performance(task_name, duration)
            else:
                logger.info(f"Execution block (sync) completed in {duration:.4f} seconds.")

    def _create_cache_key_from_representation(self, task_representation: Dict[str, Any]) -> Tuple[Any, ...]:
        """Creates a hashable cache key from a task representation dictionary."""
        key_elements_for_tuple = []

        # Prioritize specific, known keys for simple tuple creation
        simple_keys = ['task_type', 'language']
        for key in simple_keys:
            if key in task_representation:
                key_elements_for_tuple.append(task_representation[key])

        if 'main_keywords' in task_representation and isinstance(task_representation['main_keywords'], list):
            try:
                # Ensure all items in main_keywords are hashable, e.g. strings or numbers
                key_elements_for_tuple.append(tuple(sorted(task_representation['main_keywords'])))
            except TypeError:
                 # If main_keywords contains unhashable items, convert them to string
                key_elements_for_tuple.append(tuple(sorted(map(str, task_representation['main_keywords']))))


        # For more complex or less predictable parts of the representation, serialize and hash
        # This is a fallback if direct tuple creation isn't robust enough for all cases
        # For this simulation, we'll primarily rely on the selected elements above.
        # If the representation is very dynamic, hashing a sorted JSON string of it is safer.

        # Fallback: if key_elements_for_tuple is empty or too generic, use a hash of more content
        if not key_elements_for_tuple or len(key_elements_for_tuple) < 2 : # Example condition for using hash
            try:
                # Create a consistent string representation for hashing
                # Only include serializable parts
                serializable_dict = {
                    k: v for k, v in task_representation.items()
                    if isinstance(v, (str, int, float, bool, tuple, list)) # Allow lists here for JSON
                }
                # Convert lists to sorted tuples of strings to ensure hash consistency for lists
                for k, v in serializable_dict.items():
                    if isinstance(v, list):
                        serializable_dict[k] = tuple(sorted(map(str, v)))

                json_str = json.dumps(serializable_dict, sort_keys=True)
                # Return a single-element tuple containing the hash
                return (hashlib.sha256(json_str.encode('utf-8')).hexdigest(),)
            except Exception as e:
                logger.error(f"Error creating robust cache key from representation: {e}. Caching will be skipped for this item.")
                self.cache_stats['errors'] = self.cache_stats.get('errors', 0) + 1
                return (f"error_key_{uuid.uuid4()}",) # Unique key to prevent collision and indicate error

        return tuple(key_elements_for_tuple)


    async def get_or_compute(self, task_representation: Dict[str, Any], compute_func: Callable, *args, **kwargs) -> Any:
        cache_key = self._create_cache_key_from_representation(task_representation)

        if isinstance(cache_key, tuple) and len(cache_key) == 1 and isinstance(cache_key[0], str) and cache_key[0].startswith('error_key_'):
            logger.warning(f"Skipping cache for task due to key generation error: {task_representation.get('task_type', 'N/A')}")
            # Proceed to compute without caching
        elif cache_key in self.mock_cache:
            self.cache_stats['hits'] += 1
            cached_data = self.mock_cache[cache_key]
            logger.info(f"Cache hit for task: {task_representation.get('task_type', 'N/A')} (Key Hash Hint: {str(cache_key)[:100]}). Returning cached result.")
            # Optionally: log if representation is slightly different but key matches (semantic hit)
            # if task_representation != cached_data.get('original_representation_sim'):
            #    logger.info("Note: Current task representation differs slightly from cached, but key matches (semantic hit).")
            return cached_data.get('result')

        self.cache_stats['misses'] += 1
        logger.info(f"Cache miss for task: {task_representation.get('task_type', 'N/A')} (Key Hash Hint: {str(cache_key)[:100]}). Computing...")

        if asyncio.iscoroutinefunction(compute_func):
            result = await compute_func(*args, **kwargs)
        else:
            result = await asyncio.to_thread(compute_func, *args, **kwargs)

        if not (isinstance(cache_key, tuple) and len(cache_key) == 1 and isinstance(cache_key[0], str) and cache_key[0].startswith('error_key_')):
            self.mock_cache[cache_key] = {
                'result': result,
                'timestamp': datetime.utcnow().isoformat(),
                # 'original_representation_sim': task_representation # Optional: for debugging semantic hits
            }
            logger.info(f"Result computed and cached for task: {task_representation.get('task_type', 'N/A')} (Key Hash Hint: {str(cache_key)[:100]})")
        return result

    async def optimize_task_routing(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        logger.debug(f"Simulating task routing optimization for: {task_details.get('name', 'Unnamed task')}")
        await asyncio.sleep(0.01)
        return {'optimized_route': 'default_agent_pool', 'reason': 'placeholder_logic'}

    def record_task_performance(self, task_name: str, duration: float, success: bool = True):
        if task_name not in self.request_stats:
            self.request_stats[task_name] = {'count': 0, 'total_time': 0, 'success_count': 0}
        self.request_stats[task_name]['count'] += 1
        self.request_stats[task_name]['total_time'] += duration
        if success:
            self.request_stats[task_name]['success_count'] += 1
        logger.info(f"Performance recorded for task '{task_name}': duration={duration:.4f}s, success={success}")

    async def get_performance_report(self) -> Dict[str, Any]:
        report = {'tasks': {}, 'cache': self.cache_stats.copy(), 'overall_avg_time': 0.0}
        total_time_all_tasks = 0.0
        total_count_all_tasks = 0
        for name, stats in self.request_stats.items():
            avg_time = stats['total_time'] / stats['count'] if stats['count'] > 0 else 0
            success_rate = stats['success_count'] / stats['count'] if stats['count'] > 0 else 0
            report['tasks'][name] = {
                'average_time': avg_time,
                'total_executions': stats['count'],
                'success_rate': success_rate
            }
            total_time_all_tasks += stats['total_time']
            total_count_all_tasks += stats['count']

        if total_count_all_tasks > 0:
            report['overall_avg_time'] = total_time_all_tasks / total_count_all_tasks
        return report

    async def clear_cache(self):
        self.mock_cache.clear()
        self.cache_stats = {'hits': 0, 'misses': 0, 'errors': 0}
        logger.info("Mock cache cleared.")

__all__ = ['PerformanceOptimizer']
