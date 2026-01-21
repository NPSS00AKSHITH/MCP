"""Unit tests for DAG-Based Parallel Task Planner."""

import unittest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch


class TestTaskPlanner(unittest.TestCase):
    """Test cases for DAG-based task planner."""

    def setUp(self):
        """Set up test fixtures."""
        from src.retrieval.task_planner import DAGPlanner, TaskNode, ExecutionPlan
        
        self.planner = DAGPlanner()
    
    def test_is_simple_query_true(self):
        """Test simple query detection - positive cases."""
        simple_queries = [
            "What is the budget for Q3?",
            "Find the project milestones",
            "Search for Python examples",
            "Get the sales report",
        ]
        
        for query in simple_queries:
            self.assertTrue(
                self.planner.is_simple_query(query),
                f"'{query}' should be detected as simple"
            )
    
    def test_is_simple_query_false(self):
        """Test simple query detection - negative cases (complex queries)."""
        complex_queries = [
            "Compare sales report to competitor analysis",
            "What is the difference between Q1 and Q2 revenue?",
            "Contrast the pricing strategy versus market trends",
            "Cross-reference budget with actual spending",
        ]
        
        for query in complex_queries:
            self.assertFalse(
                self.planner.is_simple_query(query),
                f"'{query}' should be detected as complex"
            )
    
    def test_create_simple_plan(self):
        """Test simple query creates single-task plan."""
        plan = self.planner._create_simple_plan("Find Python functions")
        
        self.assertEqual(len(plan.tasks), 1)
        self.assertEqual(len(plan.execution_levels), 1)
        self.assertEqual(plan.execution_levels[0], ["task_0"])
        
        task = plan.tasks["task_0"]
        self.assertEqual(task.tool_name, "search")
        self.assertEqual(len(task.dependencies), 0)
    
    def test_topological_sort_parallel_tasks(self):
        """Test topological sort groups independent tasks."""
        from src.retrieval.task_planner import TaskNode
        
        # Two independent tasks
        tasks = {
            "task_0": TaskNode("task_0", "search", {"query": "A"}, set()),
            "task_1": TaskNode("task_1", "search", {"query": "B"}, set()),
        }
        
        levels = self.planner._topological_sort(tasks)
        
        # Both should be at level 0
        self.assertEqual(len(levels), 1)
        self.assertEqual(set(levels[0]), {"task_0", "task_1"})
    
    def test_topological_sort_sequential_tasks(self):
        """Test topological sort handles dependencies."""
        from src.retrieval.task_planner import TaskNode
        
        # task_2 depends on task_0 and task_1
        tasks = {
            "task_0": TaskNode("task_0", "search", {"query": "A"}, set()),
            "task_1": TaskNode("task_1", "search", {"query": "B"}, set()),
            "task_2": TaskNode("task_2", "compare", {}, {"task_0", "task_1"}),
        }
        
        levels = self.planner._topological_sort(tasks)
        
        # Level 0: task_0, task_1 (parallel)
        # Level 1: task_2 (depends on both)
        self.assertEqual(len(levels), 2)
        self.assertEqual(set(levels[0]), {"task_0", "task_1"})
        self.assertEqual(levels[1], ["task_2"])
    
    def test_execution_plan_get_level(self):
        """Test ExecutionPlan.get_level() method."""
        from src.retrieval.task_planner import TaskNode, ExecutionPlan
        
        tasks = {
            "task_0": TaskNode("task_0", "search", {}, set()),
            "task_1": TaskNode("task_1", "rerank", {}, {"task_0"}),
        }
        plan = ExecutionPlan(
            query="test",
            tasks=tasks,
            execution_levels=[["task_0"], ["task_1"]],
        )
        
        level_0 = plan.get_level(0)
        level_1 = plan.get_level(1)
        level_2 = plan.get_level(2)
        
        self.assertEqual(len(level_0), 1)
        self.assertEqual(level_0[0].task_id, "task_0")
        self.assertEqual(len(level_1), 1)
        self.assertEqual(level_1[0].task_id, "task_1")
        self.assertEqual(len(level_2), 0)  # Out of bounds
    
    def test_plan_to_dict(self):
        """Test ExecutionPlan serialization."""
        from src.retrieval.task_planner import TaskNode, ExecutionPlan
        
        tasks = {"task_0": TaskNode("task_0", "search", {"query": "test"}, set())}
        plan = ExecutionPlan("test query", tasks, [["task_0"]])
        
        plan_dict = plan.to_dict()
        
        self.assertEqual(plan_dict["query"], "test query")
        self.assertIn("task_0", plan_dict["tasks"])
        self.assertEqual(plan_dict["total_levels"], 1)


class TestTaskPlannerAsync(unittest.TestCase):
    """Async tests for plan execution."""
    
    def test_execute_plan_parallel(self):
        """Test parallel execution of independent tasks."""
        from src.retrieval.task_planner import DAGPlanner, TaskNode, ExecutionPlan, TaskStatus
        
        planner = DAGPlanner()
        
        # Create plan with 2 parallel tasks
        tasks = {
            "task_0": TaskNode("task_0", "search", {"query": "A"}, set()),
            "task_1": TaskNode("task_1", "search", {"query": "B"}, set()),
        }
        plan = ExecutionPlan("compare A and B", tasks, [["task_0", "task_1"]])
        
        # Mock tool executor
        async def mock_executor(tool_name: str, args: dict):
            await asyncio.sleep(0.01)  # Simulate work
            return {"tool": tool_name, "args": args, "result": "success"}
        
        # Execute plan
        results = asyncio.run(planner.execute_plan(plan, mock_executor))
        
        # Both tasks should have results
        self.assertEqual(len(results), 2)
        self.assertIn("task_0", results)
        self.assertIn("task_1", results)
        
        # Both tasks should be completed
        self.assertEqual(tasks["task_0"].status, TaskStatus.COMPLETED)
        self.assertEqual(tasks["task_1"].status, TaskStatus.COMPLETED)


class TestTaskPlannerWithLLM(unittest.TestCase):
    """Tests for LLM-based query decomposition."""
    
    @patch("src.retrieval.task_planner.get_llm_client")
    def test_decompose_query_llm(self, mock_get_llm):
        """Test LLM-based query decomposition."""
        from src.retrieval.task_planner import DAGPlanner
        
        mock_llm = MagicMock()
        mock_llm.generate_text.return_value = '''
        {
            "tasks": [
                {"id": "task_0", "tool": "search", "query": "sales report", "depends_on": []},
                {"id": "task_1", "tool": "search", "query": "competitor analysis", "depends_on": []}
            ]
        }
        '''
        mock_get_llm.return_value = mock_llm
        
        planner = DAGPlanner(llm_client=mock_llm)
        
        tasks = planner._decompose_query_llm(
            "Compare sales report to competitor analysis",
            ["search", "rerank", "compare_documents"]
        )
        
        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0]["tool"], "search")
        self.assertEqual(tasks[1]["tool"], "search")


if __name__ == "__main__":
    unittest.main()
