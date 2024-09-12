from langchain_core.memory import BaseMemory
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.pydantic_v1 import Field
from typing import List
from datetime import datetime

class EnhancedInMemoryHistory(BaseMemory):
    """
    Custom memory class for storing chat history with timestamps.
    Attributes:
        messages (List[BaseMessage]): A list of messages exchanged between the user and the bot.
        timestamps (List[datetime]): A list of timestamps when the messages were exchanged.
    """

    messages: List[BaseMessage] = Field(default_factory=list)
    timestamps: List[datetime] = Field(default_factory=list)

    @property
    def memory_variables(self):
        """Returns a list of memory variables (history) used in the conversation chain."""
        return ["history"]

    def add_messages(self, messages: List[BaseMessage]):
        """
        Adds new messages to the memory and timestamps them.
        Parameters:
            messages (List[BaseMessage]): A list of messages to add to the memory.
        """
        current_time = datetime.now()
        self.messages.extend(messages)
        self.timestamps.extend([current_time] * len(messages))

    def get_recent_messages(self, limit: int = 5):
        """
        Retrieves the most recent messages.
        Parameters:
            limit (int): Number of recent messages to retrieve (default is 5).
        Returns:
            List[BaseMessage]: A list of the most recent messages.
        """
        return self.messages[-limit:]

    def clear(self):
        """Clears all messages and timestamps from the memory."""
        self.messages = []
        self.timestamps = []

    def load_memory_variables(self, inputs: dict):
        """
        Loads memory variables for the conversation chain.
        Parameters:
            inputs (dict): Input data for the conversation chain.
        Returns:
            dict: A dictionary with the conversation history.
        """
        return {"history": "\n".join([msg.content for msg in self.messages])}

    def save_context(self, inputs: dict, outputs: str):
        """
        Saves the context of the conversation by storing the user input and bot output.
        Parameters:
            inputs (dict): The user input.
            outputs (str): The bot's response.
        """
        self.add_messages([HumanMessage(content=inputs['input']), AIMessage(content=str(outputs))])

    def clear_memory(self):
        """Clears the memory (alias for the clear function)."""
        self.clear()

# Store sessions in a dictionary
store = {}

def get_by_session_id(session_id: str) -> EnhancedInMemoryHistory:
    """
    Retrieves or creates a new memory session by session ID.
    Parameters:
        session_id (str): The session ID for the chat session.
    Returns:
        EnhancedInMemoryHistory: The memory object for the session.
    """
    if session_id not in store:
        store[session_id] = EnhancedInMemoryHistory()
    return store[session_id]
