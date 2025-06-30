import json
import unittest
import os 
from langchain_openai import ChatOpenAI
from src.config import load_configuration
from src.graph import build_graph

def save_report_to_file(report: dict):
    """
    Saves the final report to a JSON file in the 'output' directory.
    
    Args:
        report: The dictionary containing the final report.
    """
    try:
        # Ensure the output directory exists
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        
        file_path = os.path.join(output_dir, "report.json")
        
        # Write the report to the file with pretty printing
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
            
        print(f"\n📄 Report successfully saved to: {file_path}")
    except Exception as e:
        print(f"\n❌ ERROR: Could not save report to file. Reason: {e}")


def run_agent():
    """
    Main function to orchestrate the agent's execution.
    """
    try:
        print("🚀 --- Initializing Business Analyst Agent --- 🚀")
        
        llm_config = load_configuration()
        llm_client = ChatOpenAI(
            model=llm_config["model_name"],
            api_key=llm_config["api_key"],
            base_url=llm_config["base_url"],
            temperature=0.1
        )

        agent_app = build_graph(llm_client)
        
        sample_input = {
            "today_data": {"sales": 1200, "costs": 500, "customers": 20},
            "yesterday_data": {"sales": 1000, "costs": 300, "customers": 18},
        }
        
        print("\n▶️ --- Running Agent with Sample Data ---")
        final_state = agent_app.invoke(sample_input)
        
        print("\n\n--- ✅ AGENT RUN COMPLETE ---")
        final_report = final_state.get('report')
        
        if final_report:
            print("Final Report (Terminal Output):")
            print(json.dumps(final_report, indent=4))
            # Save the final report to a dedicated file
            save_report_to_file(final_report)
        else:
            print("No report was generated.")
        
    except ValueError as e:
        print(f"\n❌ ERROR: A configuration error occurred: {e}")
    except Exception as e:
        print(f"\n❌ ERROR: An unexpected error occurred: {e}")

def run_tests():
    """Runs the unit test suite."""
    print("\n\n🔍 --- Executing Validation Suite ---")
    loader = unittest.TestLoader()
    suite = loader.discover('tests')
    runner = unittest.TextTestRunner()
    runner.run(suite)

if __name__ == "__main__":
    run_agent()
    run_tests()
