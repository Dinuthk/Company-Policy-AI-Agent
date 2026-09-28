from google.genai import types

from app.rag.retrieval import retrieve_policy_chunks
from app.rag.vector_store import collection


# --------------------------------------------------
# Tool implementations
# --------------------------------------------------

def search_company_policy(query: str):

    matches = retrieve_policy_chunks(
        question=query,
        top_k=4
    )

    return [
        {
            "text": match["text"],
            "source": match["source"],
            "page": match["page"]
        }
        for match in matches
    ]


def list_available_policies():

    data = collection.get(
        include=["metadatas"]
    )

    policies = {
        metadata["source"]
        for metadata in data["metadatas"]
    }

    return sorted(policies)


def calculate_remaining_leave(
    annual_entitlement: float,
    used_days: float,
    carried_forward: float
):

    total_available = annual_entitlement + carried_forward

    remaining = total_available - used_days

    return {
        "annual_entitlement": annual_entitlement,
        "carried_forward": carried_forward,
        "used_days": used_days,
        "total_available": total_available,
        "remaining_leave": remaining
    }


# --------------------------------------------------
# Tool definitions (JSON schema)
# --------------------------------------------------

AGENT_TOOLS = [

    {
        "name": "search_company_policy",
        "description": (
            "Search the uploaded company policy documents. "
            "Use this tool whenever the user asks about company rules, "
            "policies, leave, remote work, security, expenses, conduct, "
            "training, or other information that may be in company documents."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The policy question or information to search for. "
                        "Rewrite follow-up questions into a standalone query."
                    )
                }
            },
            "required": ["query"],
            "additionalProperties": False
        }
    },

    {
        "name": "list_available_policies",
        "description": (
            "Return the list of company policy documents "
            "currently available in the knowledge base."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        }
    },

    {
        "name": "calculate_remaining_leave",
        "description": (
            "Calculate an employee's remaining annual leave. "
            "Use this after determining the annual leave entitlement "
            "from company policy when necessary."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "annual_entitlement": {
                    "type": "number",
                    "description": "Annual leave entitlement in days."
                },
                "used_days": {
                    "type": "number",
                    "description": "Number of leave days already used."
                },
                "carried_forward": {
                    "type": "number",
                    "description": (
                        "Unused leave days carried forward from the previous year."
                    )
                }
            },
            "required": [
                "annual_entitlement",
                "used_days",
                "carried_forward"
            ],
            "additionalProperties": False
        }
    }

]


# Gemini function declarations built from the definitions above
GEMINI_TOOLS = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name=tool["name"],
            description=tool["description"],
            parameters_json_schema=tool["parameters"]
        )
        for tool in AGENT_TOOLS
    ]
)


# --------------------------------------------------
# Dispatcher
# --------------------------------------------------

def execute_tool(
    tool_name: str,
    arguments: dict
):

    if tool_name == "search_company_policy":

        return search_company_policy(
            query=arguments["query"]
        )

    elif tool_name == "list_available_policies":

        return list_available_policies()

    elif tool_name == "calculate_remaining_leave":

        return calculate_remaining_leave(
            annual_entitlement=arguments["annual_entitlement"],
            used_days=arguments["used_days"],
            carried_forward=arguments["carried_forward"]
        )

    else:

        return {
            "error": f"Unknown tool: {tool_name}"
        }
