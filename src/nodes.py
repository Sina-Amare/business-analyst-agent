import json
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from .agent_state import AgentState, Report # Relative import

def processing_node(state: AgentState) -> Dict[str, Any]:
    """Calculates key business metrics from the raw daily data."""
    print("--- 🧠 NODE: Processing Metrics ---")
    today = state["today_data"]
    yesterday = state["yesterday_data"]
    
    profit = today["sales"] - today["costs"]
    
    revenue_change_pct = ((today["sales"] - yesterday["sales"]) / yesterday["sales"]) * 100 if yesterday["sales"] > 0 else 0.0
    cost_change_pct = ((today["costs"] - yesterday["costs"]) / yesterday["costs"]) * 100 if yesterday["costs"] > 0 else 0.0
    
    cac_today = today["costs"] / today["customers"] if today["customers"] > 0 else float('inf')
    cac_yesterday = yesterday["costs"] / yesterday["customers"] if yesterday["customers"] > 0 else float('inf')
    
    cac_increase_alert = False
    if cac_yesterday != float('inf') and cac_today > (cac_yesterday * 1.20):
        cac_increase_alert = True
    
    metrics = {
        "profit": round(profit, 2),
        "revenue_change_pct": round(revenue_change_pct, 2),
        "cost_change_pct": round(cost_change_pct, 2),
        "cac_today": round(cac_today, 2),
        "cac_yesterday": round(cac_yesterday, 2),
        "cac_increase_alert": cac_increase_alert,
    }
    
    print(f"📊 Calculated Metrics: {json.dumps(metrics, indent=2)}")
    return {"processed_metrics": metrics}


def recommendation_node(state: AgentState, llm_client: ChatOpenAI) -> Dict[str, Any]:
    """Generates an AI-driven report based on the processed metrics."""
    print("--- ✍️ NODE: Generating Recommendations ---")
    metrics = state["processed_metrics"]

    structured_llm = llm_client.with_structured_output(Report)

    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         """You are a professional business analyst AI. Your task is to provide a concise, actionable daily report.
         Analyze the provided metrics and generate a structured report covering:
         1. Profit status.
         2. Critical alerts (especially for 'cac_increase_alert').
         3. Actionable recommendations based on the data.
         Be direct and clear for a busy audience."""),
        ("human", "Today's key metrics:\n```json\n{metrics_json}\n```")
    ])

    chain = prompt | structured_llm
    # The output of a structured chain is a dictionary-like object.
    # We explicitly parse it into our Pydantic model for validation and type safety.
    raw_output = chain.invoke({"metrics_json": json.dumps(metrics)})
    report_object = Report.parse_obj(raw_output)
    
    report_dict = report_object.dict()
    print(f"📄 Generated Report: {json.dumps(report_dict, indent=2)}")
    
    return {"report": report_dict}
