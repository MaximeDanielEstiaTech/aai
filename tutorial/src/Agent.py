from inspect import signature
from langchain.schema import BaseMessage, HumanMessage, SystemMessage, AIMessage
import re
from typing import List, Union
from langchain_core.agents import AgentAction, AgentFinish, AgentActionMessageLog
from langchain.agents.chat.prompt import FORMAT_INSTRUCTIONS
import ast
from IPython.display import clear_output
from langchain_core.messages import ToolMessage
import traceback
import sys

from langchain_ollama import ChatOllama
from langchain_core.tools import Tool
from .Chatbot import Chatbot
import time


STOP_PHRASE = "Observation:"

class Agent(Chatbot):

    tools: List[Tool]
    
    def __init__(self, llm: ChatOllama, tools: List[Tool], history: List[BaseMessage] = []):
        """Initialize the chatbot with an LLM, tools, and an optional history."""
        super().__init__(llm, history)
        self.tools = tools


    def invoke(self, prompt:str) -> None:
        """Run the chatbot in interactive mode."""
        self.history.append(HumanMessage(content=prompt))
        clear_output(wait=True)
        self.print_history()
        stop = False
        while not stop:
            ai_message = self.streamUntil(STOP_PHRASE)
            self.history.append(ai_message)
            clear_output(wait=True)
            self.print_history()
            try:
                action = self.parse_action(ai_message)
                if isinstance(action, AgentAction):
                    tool_message = self.call_tool(action)
                    tool_message.pretty_print()
                    self.history.append(tool_message)
                if isinstance(action, AgentFinish):
                    stop = True
            except SyntaxError as e:
                system_message = SystemMessage(content=str(e))
                system_message.pretty_print()
                self.history.append(system_message)
            except Exception as e:
                traceback.print_exc()
                sys.exit()


    def streamUntil(self, stop: str) -> AIMessage:
        """Stream the LLM output until the stop phrase is found."""
        ai_message = AIMessage(content="")
        ai_message.pretty_print()
        print()
        for chunk in self.llm.stream(self.history):
            ai_message.content += chunk.content
            # Check for the stop phrase in the last part of the message
            start_ind = ai_message.content.find(stop, len(ai_message.content) - len(chunk.content) - len(stop))
            if start_ind != -1:
                end_ind = start_ind + len(stop)
                len_without_chunk = len(ai_message.content) - len(chunk.content)
                i = end_ind - len_without_chunk
                if i > 0:
                    print(chunk.content[:i], end="")
                ai_message.content = ai_message.content[:end_ind] 
                #sys.exit()
                break
            print(chunk.content, end="")
        print()
        return self.sanitize(ai_message)


    def call_tool(self, action: AgentAction) -> ToolMessage:
        """Call the specified tool with the given action input."""
        tool = next((t for t in self.tools if t.name == action.tool), None)
        if not tool:
            return ToolMessage(content=f"Error: Tool '{action.tool}' does not exist.", tool_call_id="unknown_tool")
        result = None
        sig = signature(tool.func)
        if len(sig.parameters):
            action_input = action.tool_input
            if not isinstance(action_input, dict):
                param_name = next(iter(sig.parameters))
                action_input = {param_name: action_input}
            result = tool.func(**action_input)
        else:
            result = tool.func()
        return ToolMessage(content=f"{result}", tool_call_id=tool.func.__name__)
    
    
    def parse_action(self, response:AIMessage) -> Union[AgentAction, AgentActionMessageLog, AgentFinish]:
        """Parse the action from the LLM output text and return an AgentAction or AgentFinish object."""
        FINAL_ANSWER_ACTION = "Final Answer:"
        pattern = re.compile(r"^.*?`{3}(?:json)?\n?(.*?)`{3}.*?$", re.DOTALL)
        includes_answer = FINAL_ANSWER_ACTION in response.content
        try:
            found = pattern.search(response.content)
            if not found:
                raise ValueError("[SyntaxError] Action not found. Did you use the correct format?")
            action = found.group(1)
            ans = ast.literal_eval(action.strip())
            return AgentAction(ans["action"], ans.get("action_input", {}), response.content)
        except Exception as e:
            if not includes_answer:
                #traceback.print_exc()
                raise SyntaxError("""[SyntaxError] 
- If an action is needed, use the following format:
  Action: 
  ```
  {{
    "action": "tool_name",
    "action_input": "your input here"
  }}
  ```
- If you want to give your final answer, use the following format:
  Final Answer: 
  ```
  your final answer here
  ```
""")
            output = response.content.split(FINAL_ANSWER_ACTION)[-1].strip()
            return AgentFinish({"output": output}, response.content)
