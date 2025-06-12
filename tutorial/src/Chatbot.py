from typing import List
from langchain.schema import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from IPython.display import clear_output
import re
import time

class Chatbot:
    llm: ChatOllama
    history: List[BaseMessage]

    def __init__(self, llm: ChatOllama, history: List[BaseMessage] = []):
        """Initialize the chatbot with an LLM and an optional history."""
        self.llm = llm
        self.history = history
    
    def clear_history(self) -> None:
        """Clear the chatbot's history."""
        self.history = []

    def invoke(self, prompt:str) -> AIMessage:
        """Run the chatbot with the current history."""
        self.history.append(HumanMessage(content=prompt))
        clear_output(wait=True)
        self.print_history()
        ai_message = AIMessage(content="")
        ai_message.pretty_print()
        print()
        for chunk in self.llm.stream(self.history):
            print(chunk.content, end="")
            ai_message.content += chunk.content
        print()
        self.history.append(self.sanitize(ai_message))
        return ai_message
        
    
    def interact(self) -> None:
        """Run the chatbot in interactive mode."""
        while True: 
            prompt = input("Prompt (Enter 'stop' to exit)")
            if prompt == "stop": 
                break
            self.invoke(prompt)


    def print_history(self) -> None:
        """Pretty print the chatbot's history."""
        for message in self.history:
            message.pretty_print()
            time.sleep(0.1)

    def sanitize(self, message: AIMessage) -> AIMessage:
        """Remove <think> tags and extra whitespace from the message content."""
        new_message_content = re.sub(r"<think>.*?</think>", "", message.content, flags=re.DOTALL).strip()
        return AIMessage(content=new_message_content)
