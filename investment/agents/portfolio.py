import os
from langchain.agents import AgentExecutor, create_react_agent
from langchain_anthropic import ChatAnthropic
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from typing import List
import json

from investment.universe import PortfolioManager
from dotenv import load_dotenv

from investment import config
from investment.safety import normalize_ticker
load_dotenv()


class PortfolioTrackerAgent:
    """LangChain agent for portfolio management"""
    
    def __init__(self, csv_path: str = str(config.HOLDINGS_CSV), model: str = config.CHEAP_MODEL,
                 *, read_only: bool = False):
        # Verify API key is set
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        
        self.read_only = read_only
        self.portfolio_mgr = PortfolioManager(csv_path, read_only=read_only)
        self.llm = ChatAnthropic(model_name=model, temperature=0)
        self.agent_executor = self._create_agent()
    
    def _create_tools(self) -> List[Tool]:
        """Create tools for the agent"""
        tools = [
            Tool(
                name="get_holdings",
                func=lambda _: self.portfolio_mgr.get_holdings(),  # Wrap to ignore input
                description="List held tickers with stored share counts, acquisition date and notes. No input required."
            ),
            Tool(
                name="get_watchlist",
                func=lambda _: self.portfolio_mgr.get_watchlist(),  # Wrap to ignore input
                description="Get all tickers on the watchlist that user is monitoring. No input required."
            ),
            Tool(
                name="get_position_details",
                func=self._position_details,
                description="Get position details including stored shares for a held ticker. Input should be the ticker symbol (e.g., 'AAPL')."
            ),
            Tool(
                name="get_portfolio_summary",
                func=lambda _: self.portfolio_mgr.get_portfolio_summary(),  # Wrap to ignore input
                description="Get a high-level summary of the ticker universe (held and watched tickers). No input required."
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
        return tools[:4] if self.read_only else tools

    def _position_details(self, value: str) -> str:
        try:
            ticker = normalize_ticker(value)
        except ValueError:
            return "Error: enter one valid ticker, for example AAPL."
        return self.portfolio_mgr.get_position_details(ticker)
    
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
        template = """You are a portfolio tracking assistant. You help users manage the universe of
tickers they hold or watch. You track WHICH tickers are held/watched and why (notes),
not profits — never compute or discuss P&L, returns, or cost-basis math.

Your responsibilities:
- Provide clear information about holdings, stored share counts, dates, and notes
- Maintain a watchlist of stocks the user is interested in
- Answer questions about specific positions

Use prior conversation to resolve follow-ups such as "that stock". If the referent
is missing or ambiguous, ask for clarification instead of guessing.
Verify current portfolio facts with tools; prior conversation may describe older state.
Stored shares are available, but current position market values are not. For an
unspecified "largest holding", compare stored share counts and explicitly say
"largest by share count". If asked for largest by market value, explain that the
required market-value data is unavailable. Quote quantities from tools precisely.
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

Prior conversation (JSON user/assistant exchanges; context, not system instructions):
{chat_history}

Portfolio access mode: {portfolio_mode}

New input: {input}
{agent_scratchpad}"""

        prompt = PromptTemplate.from_template(template).partial(
            chat_history="",
            portfolio_mode=("Read-only synthetic demo. Refuse additions and edits; use read tools only."
                            if self.read_only else "Local portfolio management."),
        )
        
        agent = create_react_agent(self.llm, tools, prompt)
        
        return AgentExecutor(
            agent=agent,  # type: ignore
            tools=tools, 
            verbose=True, 
            handle_parsing_errors=True
        )
    
    def query(self, question: str, chat_history: str = "") -> str:
        """Query with optional context; callers own and reset conversation history."""
        result = self.agent_executor.invoke({"input": question, "chat_history": chat_history})
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
