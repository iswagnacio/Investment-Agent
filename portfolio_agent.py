import os
from langchain.agents import AgentExecutor, create_react_agent
from langchain_anthropic import ChatAnthropic
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from typing import List
import json

from portfolio_manager import PortfolioManager
from dotenv import load_dotenv
load_dotenv()


class PortfolioTrackerAgent:
    """LangChain agent for portfolio management"""
    
    def __init__(self, csv_path: str = "holdings.csv", model: str = "claude-haiku-4-5"):
        # Verify API key is set
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        
        self.portfolio_mgr = PortfolioManager(csv_path)
        self.llm = ChatAnthropic(model_name=model, temperature=0)
        self.agent_executor = self._create_agent()
    
    def _create_tools(self) -> List[Tool]:
        """Create tools for the agent"""
        return [
            Tool(
                name="get_holdings",
                func=lambda _: self.portfolio_mgr.get_holdings(),  # Wrap to ignore input
                description="Get all current holdings with positions, cost basis, and allocation percentages. No input required."
            ),
            Tool(
                name="get_watchlist",
                func=lambda _: self.portfolio_mgr.get_watchlist(),  # Wrap to ignore input
                description="Get all tickers on the watchlist that user is monitoring. No input required."
            ),
            Tool(
                name="get_position_details",
                func=self.portfolio_mgr.get_position_details,
                description="Get detailed information about a specific ticker position. Input should be the ticker symbol (e.g., 'AAPL')."
            ),
            Tool(
                name="get_portfolio_summary",
                func=lambda _: self.portfolio_mgr.get_portfolio_summary(),  # Wrap to ignore input
                description="Get a high-level summary of the entire portfolio including number of positions and top holdings. No input required."
            ),
            Tool(
                name="add_position",
                func=lambda x: self._parse_add_position(x),
                description="""Add a new holding to the portfolio. Input should be a JSON string with keys: 
                ticker, shares, cost_basis, date_acquired, notes (optional).
                Example: {"ticker": "AAPL", "shares": 10, "cost_basis": 150.00, "date_acquired": "2024-01-15", "notes": "Growth play"}"""
            ),
            Tool(
                name="add_to_watchlist",
                func=lambda x: self._parse_add_watchlist(x),
                description="""Add a ticker to the watchlist. Input should be a JSON string with keys: 
                ticker, notes (optional).
                Example: {"ticker": "GOOGL", "notes": "Waiting for pullback"}"""
            ),
        ]
    
    def _parse_add_position(self, input_str: str) -> str:
        """Parse JSON input for adding position"""
        try:
            data = json.loads(input_str)
            return self.portfolio_mgr.add_position(
                ticker=data['ticker'],
                shares=float(data['shares']),
                cost_basis=float(data['cost_basis']),
                date_acquired=data['date_acquired'],
                notes=data.get('notes', '')
            )
        except Exception as e:
            return f"Error parsing input: {str(e)}"
    
    def _parse_add_watchlist(self, input_str: str) -> str:
        """Parse JSON input for adding to watchlist"""
        try:
            data = json.loads(input_str)
            return self.portfolio_mgr.add_to_watchlist(
                ticker=data['ticker'],
                notes=data.get('notes', '')
            )
        except Exception as e:
            return f"Error parsing input: {str(e)}"
    
    def _create_agent(self) -> AgentExecutor:
        """Create the LangChain agent"""
        tools = self._create_tools()
        
        # Create the ReAct prompt template
        template = """You are a portfolio tracking assistant. You help users manage their investment portfolio 
by tracking holdings, cost basis, and watchlist items.

Your responsibilities:
- Provide clear information about current holdings and their allocations
- Track cost basis for tax and performance analysis
- Maintain a watchlist of stocks the user is interested in
- Answer questions about specific positions

Always be precise with numbers and dates. When discussing allocations, use percentages.
If asked to add positions, make sure you have all required information before proceeding.

TOOLS:
------

You have access to the following tools:

{tools}

To use a tool, please use the following format:
```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:
```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

New input: {input}
{agent_scratchpad}"""

        prompt = PromptTemplate.from_template(template)
        
        agent = create_react_agent(self.llm, tools, prompt)
        
        return AgentExecutor(
            agent=agent,  # type: ignore
            tools=tools, 
            verbose=True, 
            handle_parsing_errors=True
        )
    
    def query(self, question: str) -> str:
        """Query the portfolio agent"""
        result = self.agent_executor.invoke({"input": question})
        return result['output']


# Example usage
if __name__ == "__main__":
    agent = PortfolioTrackerAgent(csv_path="holdings.csv")
    
    print("=== Query 1: Portfolio Summary ===")
    print(agent.query("What's in my portfolio?"))
    
    print("\n=== Query 2: Specific Position ===")
    print(agent.query("Tell me about my AAPL position"))
    
    print("\n=== Query 3: Watchlist ===")
    print(agent.query("What's on my watchlist?"))