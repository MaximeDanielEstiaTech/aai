from inspect import signature
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
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

class Agent(Chatbot):

    tools: List[Tool]
    
    def __init__(self, llm: ChatOllama, tools: List[Tool], history: List[BaseMessage] = []):
        """Initialize the chatbot with an LLM, tools, and an optional history."""
        super().__init__(llm, history)
        self.llm = llm.bind(stop=["Observation:"])
        self.tools = tools

    def invoke(self, prompt:str) -> None:
        """Run the chatbot in interactive mode."""
        self.history.append(HumanMessage(content=prompt))
        clear_output(wait=True)
        self.pretty_print()
        stop = False
        while not stop:
            ai_pre_action_message = self.llm.invoke(self.history)
            self.history.append(ai_pre_action_message)
            try:
                action = self.parse_action(ai_pre_action_message.content)
                if isinstance(action, AgentAction):
                    tool_message = self.call_tool(action)
                    self.history.append(tool_message)
                if isinstance(action, AgentFinish):
                    stop = True
            except SyntaxError as e:
                self.history.append(SystemMessage(content=str(e)))
            except Exception as e:
                traceback.print_exc()
                sys.exit()
            clear_output(wait=True)
            self.pretty_print()


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
        return ToolMessage(content=f"Observation: {result}", tool_call_id=tool.func.__name__)
    
    
    def parse_action(self, text:str) -> Union[AgentAction, AgentActionMessageLog, AgentFinish]:
        """Parse the action from the LLM output text and return an AgentAction or AgentFinish object."""
        FINAL_ANSWER_ACTION = "Final Answer:"
        pattern = re.compile(r"^.*?`{3}(?:json)?\n?(.*?)`{3}.*?$", re.DOTALL)
        includes_answer = FINAL_ANSWER_ACTION in text
        try:
            found = pattern.search(text)
            if not found:
                raise ValueError("action not found.")
            action = found.group(1)
            response = ast.literal_eval(action.strip())
            return AgentAction(response["action"], response.get("action_input", {}), text)
        except Exception as e:
            if not includes_answer:
                raise SyntaxError("Reminder to always use the exact characters `Final Answer:` when responding.")
            output = text.split(FINAL_ANSWER_ACTION)[-1].strip()
            return AgentFinish({"output": output}, text)
