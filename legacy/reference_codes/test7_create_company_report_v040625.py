import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Required Pydantic Model Definition (as used by the function)
# ---------------------------------------------------------------------------

class MetaSummary(BaseModel):
    """Overall summary synthesizing information from multiple sources."""
    company_name: str = Field(..., description="Name of the company being summarized")
    summary: str = Field(..., description="Comprehensive summary of the company")
    information_gaps: List[str] = Field(default_factory=list, description="Gaps in information that require further research")
    conflicting_information: List[str] = Field(default_factory=list, description="Areas where sources provide conflicting information")
    is_comprehensive: bool = Field(False, description="Indicates if the summary comprehensively covers the company")
    has_information_gaps: bool = Field(False, description="Indicates if there are significant information gaps")
    has_conflicting_information: bool = Field(False, description="Indicates if there is conflicting information from sources")
    is_high_confidence: bool = Field(False, description="Indicates if the information has high confidence level")
    requires_additional_research: bool = Field(False, description="Indicates if additional research is needed")

# ---------------------------------------------------------------------------
# JSON Report Generation Function
# ---------------------------------------------------------------------------

def create_company_report(consolidated_info: Dict[str, Any], meta_summary: MetaSummary, company_name: str, github_stats: Dict[str, Any]) -> str:
    """
    Create a formatted JSON report from consolidated company information.

    Args:
        consolidated_info: Consolidated company information dictionary containing
                           keys like 'company_data', 'leadership_team',
                           'product_offerings', 'funding_data', 'competitors', etc.
        meta_summary: Complete MetaSummary object (or dict) with analysis.
        company_name: Name of the company.
        github_stats: Dictionary containing GitHub repository statistics.

    Returns:
        Formatted JSON report string.
    """

    # Handle potential input type for meta_summary (Pydantic model vs dict)
    if hasattr(meta_summary, 'model_dump'):
        meta_summary_dict = meta_summary.model_dump()
    else:
        # Assume it's already a dictionary-like object
        meta_summary_dict = meta_summary

    # Initialize company_data with required fields, ensuring defaults
    company_data = consolidated_info.get("company_data", {}).copy() # Use copy to avoid modifying original
    if not company_data: # If company_data was missing or empty
        company_data = {"name": company_name}

    # Ensure company name is present
    if "name" not in company_data or not company_data["name"]:
        company_data["name"] = company_name

    # Define required fields and their defaults for company_data
    required_company_fields = {
        "industry_tags": [],
        "founded": "",
        "headquarters": "",
        "website": "",
        "total_funding": "",
        "latest_funding_round": "",
        # Adding fields potentially expected by the dashboard structure
        "monthly_downloads": "",
        "organizations_using": "",
        "fortune_500_clients": "",
        "business_model": "",
        "company_size": "",
        "description": "",
        "has_open_source_products": False, # Example boolean field
        "is_ai_focused": False,           # Example boolean field
        "is_startup": False               # Example boolean field
    }

    for field, default_value in required_company_fields.items():
        company_data.setdefault(field, default_value) # setdefault is concise

    # Process funding_data to ensure structure and defaults
    processed_funding_data = []
    raw_funding_data = consolidated_info.get("funding_data", [])
    if raw_funding_data: # Check if there is funding data to process
        for round_data in raw_funding_data:
            funding_round = round_data.copy() # Use copy

            # Ensure required fields exist with defaults
            funding_round.setdefault("round", "")
            funding_round.setdefault("date", "")
            funding_round.setdefault("amount", "")
            funding_round.setdefault("lead_investor", "")
            funding_round.setdefault("source_url", "")

            # Handle investors list carefully
            investors = funding_round.get("investors")
            if investors is None:
                funding_round["investors"] = []
            elif isinstance(investors, str):
                # Convert comma-separated string to list
                funding_round["investors"] = [inv.strip() for inv in investors.split(',') if inv.strip()]
            # If it's already a list, keep it as is

            processed_funding_data.append(funding_round)

    # Determine latest funding round for company_data if not already set
    if not company_data.get("latest_funding_round") and processed_funding_data:
        # Sort rounds by date (best effort parsing)
        def get_sortable_date(r):
            date_str = r.get("date", "")
            try:
                # Attempt to parse common date formats
                return datetime.strptime(date_str, '%B %d, %Y') # Example: May 1, 2015
            except ValueError:
                try:
                    return datetime.strptime(date_str, '%Y-%m-%d') # Example: 2015-05-01
                except ValueError:
                     try:
                         return datetime.strptime(date_str, '%Y') # Example: 2015
                     except ValueError:
                         return datetime.min # Cannot parse, put at the beginning
            except TypeError:
                 return datetime.min # Handle None or non-string types


        sorted_rounds = sorted(processed_funding_data, key=get_sortable_date, reverse=True)
        if sorted_rounds:
            latest_round = sorted_rounds[0]
            round_name = latest_round.get('round', '')
            round_date = latest_round.get('date', '')
            # Format as "Round - Date" if both exist
            latest_round_str = f"{round_name} - {round_date}".strip(" -")
            if latest_round_str:
                 company_data["latest_funding_round"] = latest_round_str


    # --- Build the main report structure ---
    report_data = {}

    # 1. Company Data Section
    report_data["company_data"] = company_data

    # 2. Executive Summary Section
    report_data["executive_summary"] = {
        "company_summary": meta_summary_dict.get("summary", ""),
        # Replicate some fields for potential dashboard use
        "overview": meta_summary_dict.get("summary", ""),
        "information_gaps": meta_summary_dict.get("information_gaps", []),
        "confidence_level": "High" if meta_summary_dict.get("is_high_confidence", False) else "Medium",
        "additional_research_needed": meta_summary_dict.get("requires_additional_research", False),
        # Add synthetic strengths based on company_data for dashboard example
        "strengths": [
            f"{'Open source framework' if company_data.get('has_open_source_products') else 'Proprietary technology'}",
            f"{'Strong AI focus' if company_data.get('is_ai_focused') else 'Diverse technology portfolio'}",
            f"{'Innovative startup' if company_data.get('is_startup') else 'Established company'}"
        ]
    }

    # 3. Leadership Team Section
    report_data["leadership_team"] = consolidated_info.get("leadership_team", [])

    # 4. Product Offerings Section
    report_data["product_offerings"] = consolidated_info.get("product_offerings", [])

    # 5. Funding Data Section (use processed data)
    report_data["funding_data"] = processed_funding_data
    report_data["funding_analysis"] = consolidated_info.get("funding_analysis", "No specific funding analysis available.") # Default text

    # 6. Competitors Section
    report_data["competitors"] = consolidated_info.get("competitors", [])

    # 7. Recent News Section
    report_data["recent_news"] = consolidated_info.get("recent_news", [])

    # 8. Market Trends Section
    report_data["market_trends"] = consolidated_info.get("market_trends", [])

    # 9. GitHub Stats Section
    report_data["github_stats"] = github_stats if github_stats else {} # Ensure it's at least an empty dict

    # --- Add sections specifically for the dashboard structure (based on original code) ---

    # 10. Company Profile (for dashboard)
    report_data["company_profile"] = {
        "description": company_data.get("description", ""),
         # Provide example use cases if none extracted
        "use_cases": consolidated_info.get("use_cases", [
            {"title": "Primary Use Case", "description": "The company's main product offering and application.", "style": "blue"},
            {"title": "Secondary Use Case", "description": "Additional applications and use cases.", "style": "green"},
            {"title": "Industry Application", "description": "How products are used in specific industries.", "style": "purple"}
        ])
    }

    # 11. Key Technologies (for dashboard)
    report_data["key_technologies"] = consolidated_info.get("key_technologies", [
        {"category": "Primary Technology", "items": "Core technology area (e.g., AI, Vector DB)"},
        {"category": "Infrastructure", "items": "Deployment details (e.g., Cloud, On-prem)"},
        {"category": "Integration", "items": "Connectivity options (e.g., APIs, SDKs)"}
    ])

    # 12. Technical Differentiation (for dashboard)
    report_data["technical_differentiation"] = consolidated_info.get("technical_differentiation", [
        {"title": "Technology Advantage", "description": "Primary technical differentiation.", "style": "blue"},
        {"title": "Architectural Approach", "description": "Distinct architectural features.", "style": "green"}
    ])

    # 13. Market Advantages (for dashboard)
    report_data["market_advantages"] = consolidated_info.get("market_advantages", [
        {"title": "Market Position", "description": "Company's relative position in the market."}
    ])

    # 14. Strategic Direction (for dashboard)
    report_data["strategic_direction"] = consolidated_info.get("strategic_direction", [
        {"title": "Business Strategy", "description": "Current business strategy and goals."}
    ])

    # 15. Information Gaps (directly from meta_summary for dashboard)
    report_data["information_gaps"] = meta_summary_dict.get("information_gaps", [])

    # 16. Notable Customers (for dashboard)
    report_data["notable_customers"] = consolidated_info.get("notable_customers", [
        {"name": "Customer Example", "industry": "Relevant Industry", "useCase": "Example application"}
    ])

    # 17. GitHub Analysis Text (for dashboard)
    report_data["github_analysis"] = {
        "importance": consolidated_info.get("github_analysis_importance",
            "GitHub stars serve as a proxy for community interest and adoption. For open-source projects, a growing star count indicates increasing developer awareness and potential usage. The rate of star growth can also indicate momentum and market penetration compared to competitors."
        )
    }

    # --- Final Metadata ---

    # 18. Report Metadata
    timestamp = datetime.now()
    company_initial = company_name[0].upper() if company_name else "X"
    report_id = f"BI-{timestamp.strftime('%Y%m%d')}-{company_initial}-01" # Simplified ID

    report_data["report_metadata"] = {
        "report_date": timestamp.strftime("%B %d, %Y"),
        "data_confidence": meta_summary_dict.get("confidence_level", "Medium"), # Use level from summary
        "report_id": report_id,
        "generated_on": timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        "report_version": "1.0",
        "disclaimer": "This report is based on publicly available information gathered through automated processes. Information should be verified for critical decisions."
    }

    # Convert the Python dictionary to a JSON string
    return json.dumps(report_data, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Example Usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # --- 1. Create Sample Input Data ---
    sample_company_name = "ExampleTech AI"

    # Sample data that mimics the output of the information extraction phase
    sample_consolidated_info = {
        "company_data": {
            "name": "ExampleTech AI",
            "founded": "2021",
            "headquarters": "San Francisco, USA",
            "website": "https://exampletech.ai",
            "description": "ExampleTech AI provides cutting-edge AI solutions for enterprise automation.",
            "business_model": "B2B SaaS",
            "company_size": "50-100 employees",
            "total_funding": "$15M",
            "industry_tags": ["AI", "Enterprise Software", "SaaS"],
            "is_ai_focused": True,
            "is_startup": True,
            "has_open_source_products": False,
        },
        "executive_summary": { # Usually generated by another agent, but needed for completeness score
             "company_summary": "ExampleTech AI is a growing startup...",
             "strengths": ["Strong AI team", "Innovative product"],
             "information_gaps": ["Detailed pricing not found"],
             "confidence_level": "Medium",
             "additional_research_needed": True
        },
        "leadership_team": [
            {"name": "Alice CEO", "role": "CEO & Co-Founder", "background": "Ex-Google AI", "linkedin_url": "", "is_founder": True, "is_executive": True},
            {"name": "Bob CTO", "role": "CTO & Co-Founder", "background": "PhD Stanford AI", "linkedin_url": "", "is_founder": True, "is_executive": True}
        ],
        "product_offerings": [
            {"name": "AutomateFlow", "description": "AI platform for workflow automation.", "features": ["NLP processing", "Task routing"], "target_market": "Large Enterprises", "launch_date": "2022", "user_base": "Growing adoption"},
             {"name": "InsightAI", "description": "AI analytics tool.", "features": ["Predictive modeling", "Data visualization"], "target_market": "Finance Sector", "launch_date": "2023", "user_base": "Early adopters"}
        ],
        "funding_data": [
            {"round": "Seed", "date": "June 2021", "amount": "$2M", "investors": ["Seed Ventures", "Angel Investor"], "lead_investor": "Seed Ventures", "source_url": ""},
            {"round": "Series A", "date": "October 2022", "amount": "$13M", "investors": ["Growth Capital", "Seed Ventures"], "lead_investor": "Growth Capital", "source_url": "https://techcrunch.example.com/funding"}
        ],
         "funding_analysis": "ExampleTech AI has secured solid early-stage funding, indicating investor confidence. The Series A round provides runway for product development and market expansion.",
        "competitors": [
            {"name": "CompetitorX", "description": "Established player in automation.", "founded": "2015", "strengths": ["Large customer base"], "weaknesses": ["Slower AI adoption"], "market_share": "20%"},
            {"name": "AIBots Inc.", "description": "Direct AI competitor.", "founded": "2020", "strengths": ["Strong research team"], "weaknesses": ["Limited market presence"], "market_share": "5%"}
        ],
        "market_trends": [
            {"name": "Rise of Enterprise AI", "description": "Increasing demand for AI in businesses.", "impact_level": "High", "timeline": "Current", "supporting_data": ""},
             {"name": "Focus on Responsible AI", "description": "Growing importance of ethical AI development.", "impact_level": "Medium", "timeline": "Emerging", "supporting_data": ""}
        ],
        "recent_news": [
            {"date": "March 2023", "title": "ExampleTech AI Launches InsightAI", "summary": "New analytics product announced.", "url": "https://exampletech.ai/news"},
             {"date": "October 2022", "title": "ExampleTech AI Raises $13M Series A", "summary": "Funding round led by Growth Capital.", "url": "https://techcrunch.example.com/funding"}
        ],
        # Assume these dashboard-specific fields weren't explicitly extracted
        "key_technologies": None,
        "technical_differentiation": None,
        "market_advantages": None,
        "strategic_direction": None,
        "notable_customers": None,
    }

    # Sample MetaSummary (this would normally come from the meta_summary_agent)
    sample_meta_summary = MetaSummary(
        company_name=sample_company_name,
        summary="ExampleTech AI is a promising AI startup founded in 2021, specializing in enterprise automation. With $15M in total funding, led by a strong technical founding team (Alice CEO, Bob CTO), the company offers products like AutomateFlow and InsightAI. Key competitors include CompetitorX and AIBots Inc. The company is well-positioned to capitalize on the rise of enterprise AI but needs to expand its market presence.",
        information_gaps=["Detailed pricing strategy", "Specific customer case studies"],
        conflicting_information=[],
        is_comprehensive=True,
        has_information_gaps=True,
        has_conflicting_information=False,
        is_high_confidence=False, # Medium confidence due to gaps
        requires_additional_research=True
    )

    # Sample GitHub Stats (this would normally come from the GitHub API call)
    sample_github_stats = {
        "name": "AutomateFlow-SDK",
        "repository": "ExampleTechAI/AutomateFlow-SDK",
        "html_url": "https://github.com/ExampleTechAI/AutomateFlow-SDK",
        "stars": 523,
        "forks": 45,
        "created_at": "2022-01-15T10:00:00Z",
        "updated_at": "2023-11-01T12:30:00Z",
        "open_issues": 12,
        "subscribers": 30
    }

    # --- 2. Generate the JSON Report ---
    json_report_string = create_company_report(
        consolidated_info=sample_consolidated_info,
        meta_summary=sample_meta_summary,
        company_name=sample_company_name,
        github_stats=sample_github_stats
    )

    # --- 3. Print the Result ---
    print("Generated JSON Report:")
    print(json_report_string)

    # --- 4. Optional: Validate JSON ---
    try:
        json.loads(json_report_string)
        print("\nJSON is valid.")
    except json.JSONDecodeError as e:
        print(f"\nGenerated JSON is invalid: {e}")