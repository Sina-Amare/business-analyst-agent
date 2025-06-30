import unittest
from langchain_openai import ChatOpenAI

# Import functions and classes from the src directory
from src.config import load_configuration
from src.graph import build_graph
from src.nodes import processing_node
from src.agent_state import AgentState

class TestBusinessAgent(unittest.TestCase):
    """A suite of tests to validate the agent's components and logic."""

    def test_processing_node_logic(self):
        """Validates the core calculation logic of the processing node."""
        print("\n--- 🔬 RUNNING TEST: test_processing_node_logic ---")
        test_state = AgentState(
            today_data={"sales": 2000, "costs": 800, "customers": 40},
            yesterday_data={"sales": 1500, "costs": 600, "customers": 30},
            processed_metrics=None,
            report=None
        )
        result = processing_node(test_state)
        metrics = result["processed_metrics"]
        
        self.assertAlmostEqual(metrics["profit"], 1200)
        self.assertAlmostEqual(metrics["revenue_change_pct"], 33.33)
        self.assertAlmostEqual(metrics["cac_today"], 20)
        self.assertFalse(metrics["cac_increase_alert"])

    def test_cac_alert_trigger(self):
        """Ensures the CAC increase alert is triggered correctly."""
        print("\n--- 🔬 RUNNING TEST: test_cac_alert_trigger ---")
        test_state = AgentState(
            today_data={"sales": 1000, "costs": 500, "customers": 10},
            yesterday_data={"sales": 1000, "costs": 400, "customers": 10},
            processed_metrics=None, report=None
        )
        result = processing_node(test_state)
        self.assertTrue(result["processed_metrics"]["cac_increase_alert"])

    def test_full_agent_run_structure(self):
        """Performs an end-to-end integration test of the compiled graph."""
        print("\n--- 🔬 RUNNING TEST: test_full_agent_run_structure ---")
        try:
            config = load_configuration()
            llm = ChatOpenAI(
                model=config["model_name"],
                api_key=config["api_key"],
                base_url=config["base_url"],
                temperature=0
            )
            app = build_graph(llm)
            
            sample_input = {
                "today_data": {"sales": 1200, "costs": 500, "customers": 20},
                "yesterday_data": {"sales": 1000, "costs": 300, "customers": 18},
            }
            final_state = app.invoke(sample_input)
            
            self.assertIn("report", final_state)
            report = final_state["report"]
            self.assertIsInstance(report, dict)
            self.assertIn("profit_status", report)
            self.assertIn("alerts", report)
            self.assertIn("recommendations", report)

        except ValueError as e:
            self.skipTest(f"Skipping integration test due to configuration error: {e}")

if __name__ == '__main__':
    unittest.main()
