"""DAG-based task planner for parallel multi-tool execution.

Handles queries requiring multiple independent retrievals by:
1. Decomposing query into atomic tasks
2. Identifying dependencies (DAG structure)
3. Executing independent tasks in parallel
4. Merging results

Example:
    Query: "Compare sales report to competitor analysis"
    
    Plan:
        Level 0 (parallel):
            - Task A: search("sales report")
            - Task B: search("competitor analysis")
        Level 1 (depends on 0):
            - Task C: compare(results_A, results_B)
"""

import asyncio
from dataclasses import dataclass, field
from typing import List, Set, Dict, Any, Optional, Callable, Awaitable
from enum import Enum
import re

from src.server.logging import get_logger
from src.server.llm import get_llm_client

logger = get_logger(__name__)


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskNode:
    """Represents a single task in the execution DAG.
    
    Attributes:
        task_id: Unique identifier
        tool_name: MCP tool to invoke
        arguments: Arguments for tool call
        dependencies: Set of task_ids this task depends on
        status: Current execution status
        result: Result after completion (None if pending)
        error: Error message if failed
    """
    task_id: str
    tool_name: str
    arguments: Dict[str, Any]
    dependencies: Set[str] = field(default_factory=set)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "dependencies": list(self.dependencies),
            "status": self.status.value,
            "has_result": self.result is not None,
            "error": self.error,
        }


@dataclass
class ExecutionPlan:
    """Complete execution plan as a DAG.
    
    The plan is organized into "levels" where:
    - Level 0: Tasks with no dependencies (execute first, in parallel)
    - Level 1: Tasks depending only on Level 0
    - Level N: Tasks depending on earlier levels
    
    Attributes:
        query: Original user query
        tasks: Mapping of task_id to TaskNode
        execution_levels: List of lists, each containing task IDs for that level
    """
    query: str
    tasks: Dict[str, TaskNode]
    execution_levels: List[List[str]]
    
    def get_level(self, level: int) -> List[TaskNode]:
        """Get all tasks at a specific level."""
        if level >= len(self.execution_levels):
            return []
        return [self.tasks[tid] for tid in self.execution_levels[level]]
    
    def total_tasks(self) -> int:
        """Get total number of tasks."""
        return len(self.tasks)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "query": self.query,
            "tasks": {tid: t.to_dict() for tid, t in self.tasks.items()},
            "execution_levels": self.execution_levels,
            "total_levels": len(self.execution_levels),
        }


# =============================================================================
# DAG PLANNER
# =============================================================================

# Keywords suggesting query needs multi-step planning
MULTI_STEP_KEYWORDS = [
    "compare", "contrast", "between", "and", "both",
    "versus", "vs", "difference", "relationship",
    "cross-reference", "correlate", "evaluate against",
]

# Prompt template for query decomposition
DECOMPOSITION_PROMPT = """Decompose this query into atomic retrieval tasks.

Query: {query}

Available tools: {tools}

Rules:
1. Each task should be ONE retrieval operation
2. Identify if tasks can run independently (parallel) or need results from other tasks (sequential)
3. Keep the number of tasks minimal (usually 2-4 for comparison queries)

Output format (JSON):
{{
  "tasks": [
    {{"id": "task_0", "tool": "search", "query": "...", "depends_on": []}},
    {{"id": "task_1", "tool": "search", "query": "...", "depends_on": []}},
    {{"id": "task_2", "tool": "compare_documents", "depends_on": ["task_0", "task_1"]}}
  ]
}}

JSON output:"""


class DAGPlanner:
    """Plans and executes multi-task queries using DAG structure.
    
    Design Philosophy:
    - Use LLM to decompose query into tasks (planning intelligence)
    - Use graph algorithms to identify parallelism (execution efficiency)
    - Execute independent tasks concurrently (reduce latency)
    
    Example:
        >>> planner = DAGPlanner()
        >>> plan = planner.plan("Compare sales report to competitor analysis")
        >>> plan.execution_levels
        [['task_0', 'task_1'], ['task_2']]  # Level 0 parallel, Level 1 depends
    """
    
    def __init__(self, llm_client=None):
        """Initialize DAG planner.
        
        Args:
            llm_client: Optional LLM client for query decomposition.
        """
        self._llm_client = llm_client
    
    @property
    def llm(self):
        """Lazy-load LLM client."""
        if self._llm_client is None:
            self._llm_client = get_llm_client()
        return self._llm_client
    
    def is_simple_query(self, query: str) -> bool:
        """Quick heuristic: does query need multi-step planning?
        
        Args:
            query: User query string.
            
        Returns:
            True if query is simple (single retrieval), False if complex.
        """
        query_lower = query.lower()
        return not any(kw in query_lower for kw in MULTI_STEP_KEYWORDS)
    
    def plan(self, query: str, available_tools: List[str]) -> ExecutionPlan:
        """Decompose query into DAG of tasks.
        
        Steps:
        1. Use LLM to identify atomic sub-tasks
        2. For each sub-task, select appropriate tool
        3. Identify dependencies between tasks
        4. Organize into execution levels
        
        Args:
            query: User query
            available_tools: List of tool names available (from discovery)
            
        Returns:
            ExecutionPlan ready for execution
        """
        # Check for simple query - create single task plan
        if self.is_simple_query(query):
            logger.info("simple_query_detected", query_length=len(query))
            return self._create_simple_plan(query)
        
        # Use LLM to decompose complex query
        try:
            tasks = self._decompose_query_llm(query, available_tools)
        except Exception as e:
            logger.warning(
                "decomposition_failed",
                error=str(e),
                fallback="single_task",
            )
            return self._create_simple_plan(query)
        
        # Build DAG and compute execution levels
        plan = self._build_execution_plan(query, tasks)
        
        logger.info(
            "plan_created",
            query_length=len(query),
            total_tasks=plan.total_tasks(),
            execution_levels=len(plan.execution_levels),
        )
        
        return plan
    
    def _create_simple_plan(self, query: str) -> ExecutionPlan:
        """Create a single-task plan for simple queries."""
        task = TaskNode(
            task_id="task_0",
            tool_name="search",
            arguments={"query": query, "mode": "hybrid", "k": 5},
            dependencies=set(),
        )
        return ExecutionPlan(
            query=query,
            tasks={"task_0": task},
            execution_levels=[["task_0"]],
        )
    
    def _decompose_query_llm(
        self,
        query: str,
        available_tools: List[str]
    ) -> List[Dict[str, Any]]:
        """Use LLM to decompose query into tasks."""
        prompt = DECOMPOSITION_PROMPT.format(
            query=query,
            tools=", ".join(available_tools),
        )
        
        response = self.llm.generate_text(prompt)
        
        # Parse JSON from response
        # Try to extract JSON from the response
        import json
        
        # Clean up response - find JSON block
        json_match = re.search(r'\{[\s\S]*\}', response)
        if not json_match:
            raise ValueError("No JSON found in LLM response")
        
        parsed = json.loads(json_match.group())
        return parsed.get("tasks", [])
    
    def _build_execution_plan(
        self,
        query: str,
        task_defs: List[Dict[str, Any]]
    ) -> ExecutionPlan:
        """Build execution plan from task definitions.
        
        Performs topological sort to determine execution levels.
        """
        # Create TaskNodes
        tasks: Dict[str, TaskNode] = {}
        for task_def in task_defs:
            task_id = task_def.get("id", f"task_{len(tasks)}")
            tasks[task_id] = TaskNode(
                task_id=task_id,
                tool_name=task_def.get("tool", "search"),
                arguments={"query": task_def.get("query", query)},
                dependencies=set(task_def.get("depends_on", [])),
            )
        
        # Topological sort to get execution levels
        execution_levels = self._topological_sort(tasks)
        
        return ExecutionPlan(
            query=query,
            tasks=tasks,
            execution_levels=execution_levels,
        )
    
    def _topological_sort(self, tasks: Dict[str, TaskNode]) -> List[List[str]]:
        """Topological sort with level grouping.
        
        Groups tasks by their dependency level for parallel execution.
        
        Returns:
            List of lists, where each inner list contains task IDs
            that can be executed in parallel.
        """
        # Calculate in-degree for each task
        in_degree: Dict[str, int] = {tid: 0 for tid in tasks}
        for task in tasks.values():
            for dep in task.dependencies:
                if dep in tasks:  # Only count valid dependencies
                    in_degree[task.task_id] = in_degree.get(task.task_id, 0) + 1
        
        # BFS-based topological sort with levels
        levels: List[List[str]] = []
        current_level = [tid for tid, deg in in_degree.items() if deg == 0]
        
        while current_level:
            levels.append(current_level)
            next_level = []
            
            for tid in current_level:
                # Reduce in-degree of dependents
                for other_tid, other_task in tasks.items():
                    if tid in other_task.dependencies:
                        in_degree[other_tid] -= 1
                        if in_degree[other_tid] == 0:
                            next_level.append(other_tid)
            
            current_level = next_level
        
        return levels
    
    async def execute_plan(
        self,
        plan: ExecutionPlan,
        tool_executor: Callable[[str, Dict[str, Any]], Awaitable[Any]],
    ) -> Dict[str, Any]:
        """Execute the plan with parallel execution at each level.
        
        Args:
            plan: The ExecutionPlan to execute
            tool_executor: Async function that executes tool calls
            
        Returns:
            Final merged results from all tasks
        """
        results: Dict[str, Any] = {}
        
        for level_num, level_task_ids in enumerate(plan.execution_levels):
            logger.info(
                "executing_level",
                level=level_num,
                task_count=len(level_task_ids),
                parallel=True,
            )
            
            # Execute all tasks at this level in parallel
            level_results = await self._execute_level_parallel(
                plan, level_num, tool_executor
            )
            results.update(level_results)
        
        # Mark plan as complete
        logger.info(
            "plan_execution_complete",
            total_levels=len(plan.execution_levels),
            total_results=len(results),
        )
        
        return results
    
    async def _execute_level_parallel(
        self,
        plan: ExecutionPlan,
        level: int,
        tool_executor: Callable[[str, Dict[str, Any]], Awaitable[Any]],
    ) -> Dict[str, Any]:
        """Execute all tasks at a level in parallel using asyncio.gather."""
        tasks = plan.get_level(level)
        
        async def execute_task(task: TaskNode) -> tuple[str, Any]:
            """Execute a single task and return (task_id, result)."""
            task.status = TaskStatus.RUNNING
            try:
                # Inject results from dependencies if needed
                args = task.arguments.copy()
                
                result = await tool_executor(task.tool_name, args)
                task.status = TaskStatus.COMPLETED
                task.result = result
                return (task.task_id, result)
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                logger.error(
                    "task_execution_failed",
                    task_id=task.task_id,
                    error=str(e),
                )
                return (task.task_id, None)
        
        # Execute all tasks in parallel
        if tasks:
            results = await asyncio.gather(*[execute_task(t) for t in tasks])
            return {tid: result for tid, result in results}
        
        return {}


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_planner: Optional[DAGPlanner] = None


def get_dag_planner() -> DAGPlanner:
    """Get or create global DAG planner instance."""
    global _planner
    if _planner is None:
        _planner = DAGPlanner()
    return _planner
