from typing import TypedDict, Optional, List, Dict, Any
from pydantic import BaseModel, Field

class AgentState(TypedDict):
    """Defines the structured state for the agent's workflow."""
    today_data: Dict[str, float]
    yesterday_data: Dict[str, float]
    processed_metrics: Optional[Dict[str, Any]]
    report: Optional[Dict[str, Any]]

class Report(BaseModel):
    """Pydantic model for the structured LLM output."""
    profit_status: str = Field(description="A brief summary of the profit or loss status.")
    alerts: List[str] = Field(description="A list of critical warnings requiring immediate attention.")
    recommendations: List[str] = Field(description="2-3 clear, actionable recommendations for the business.")
