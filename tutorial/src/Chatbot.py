from typing import List
from langchain.schema import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from IPython.display import clear_output

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
        clear_output(wait=True)
        self.pretty_print()
        human_message = HumanMessage(content=prompt)
        human_message.pretty_print()
        self.history.append(human_message)
        ai_message = AIMessage(content="")
        ai_message.pretty_print()
        print()
        for chunk in self.llm.stream(self.history):
            print(chunk.content, end="")
            ai_message.content += chunk.content
        print()
        self.history.append(ai_message)
        return ai_message
        
    
    def interact(self) -> None:
        """Run the chatbot in interactive mode."""
        while True: 
            prompt = input("Prompt (Enter 'stop' to exit)")
            if prompt == "stop": 
                break
            self.invoke(prompt)


    def pretty_print(self) -> None:
        """Pretty print the chatbot's history."""
        for message in self.history:
            message.pretty_print()