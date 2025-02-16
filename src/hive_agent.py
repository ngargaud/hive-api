import json
import os
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables.graph import MermaidDrawMethod
from langchain_core.messages import HumanMessage, AIMessage

# in notebook
try:
    from IPython.display import Image, display
except:
    print("Not in notebook")

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    inner_thought: Annotated[list, add_messages]
    # notifications: []

class HiveAgent():
    """
    Use the Hive services to complete a task.
    The agent takes text, audio and image as parameters.
    If the text is a json then we will assume it comes from a device.
    In this case the json should contains the text, audio and image extracted informations.
    ```
    {
        text: from asr,
        language: from asr,
        voices: from voice_reco,
        faces: from face_reco
    }
    ```
    """
    def __init__(self, hive, name="HiveAgent", debug_graph=False):
        self.hive = hive
        self.name = name
        self.debug = debug_graph
        self.agent = None # graph
        self.memory = MemorySaver()
        self.subgraphs = False
        self.config = {"configurable": {"thread_id": 0, "user": "unknown"}}

    def get_memory(self):
        return self.memory

    def show_graph(self, xray=False):
        if self.agent:
            try:
                if xray:
                    # Setting xray to 1 will show the internal structure of the nested graph
                    display(Image(self.agent.get_graph(xray=1).draw_mermaid_png(draw_method=MermaidDrawMethod.API)))
                else:
                    display(Image(self.agent.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)))
            except:
                print("ERROR: show_graph() Not in notebook")


    def _parse_text(self, input):
        if input:
            if input.startswith("{"):
                try:
                    data = json.parse(input)
                except:
                    data = None
                if data:
                    return data.get("text")
        return None


    def _get_metadata(self, input, audio, image):
        if self._parse_text(input) == None:
            print("No metadata from input")
            metadata = self.hive.create_metadata(audio, image)
            metadata["text"] = input
        else:
            print("Use metadata from input")
            metadata = input
        return metadata


    def load_history_node(self, state: AgentState, config):
        user = config["configurable"].get("user", "unknown")
        print("{}.load_history_node() User {}".format(self.name, user))
        return state


    def save_history_node(self, state: AgentState, config):
        user = config["configurable"].get("user", "unknown")
        print("{}.load_history_node() User {}".format(self.name, user))
        return state


    def debug_node(self, state: AgentState, config):
        user = config["configurable"].get("user", "unknown")
        print("{}.debug_node() User {} request".format(self.name, user))
        state["inner_thought"] = AIMessage(content="The user {} say <request summary>".format(user), id=0)
        # state["messages"] = AIMessage(content="Can you provide more information?", id=0)
        return state


    def build_graph(self, subgraph=None, subname="debug_node"):
        # Define a new graph
        workflow = StateGraph(AgentState)
        subnode = self.debug_node
        if subgraph:
            self.subgraphs = True
            subnode = subgraph
            if subname == "debug_node":
                subname = "subgraph"

        # Define the two nodes we will cycle between
        workflow.add_node("{}.load_history_node".format(self.name), self.load_history_node)
        workflow.add_node("{}.save_history_node".format(self.name), self.save_history_node)
        workflow.add_node("{}.{}".format(self.name, subname), subnode)
        workflow.set_entry_point("{}.load_history_node".format(self.name))

        workflow.add_edge("{}.load_history_node".format(self.name), "{}.{}".format(self.name, subname))
        workflow.add_edge("{}.{}".format(self.name, subname), "{}.save_history_node".format(self.name))
        workflow.add_edge("{}.save_history_node".format(self.name), END)

        self.agent = workflow.compile(checkpointer=self.memory)


    def get_state(self, config, show=False):
        state = self.agent.get_state(config).values
        if show:
            print()
            print()
            print("----------------")
            print("Messages")
            for s in state.get("messages", []):
                print("{}: {}".format(type(s), s.content))
            print("----------------")
            print("Inner thought")
            for s in state.get("inner_thought", []):
                print("{}: {}".format(type(s), s.content))
            print("----------------")
            print("conversation_context: {}".format(state.get("conversation_context")))
        return state


    def get_ai_answer(self, config):
        messages = self.get_state(config, True).get("messages", [])
        if len(messages):
            last_message = messages[-1]
            if type(last_message) == AIMessage:
                return last_message
        return None


    def on_input(self, input, audio=None, image=None):
        """
        Always walk throught the full graph, exit when a response emerge.
        les subgraphs ne sont pas pris en compte dans le stream...

        Ensuite on revois la meca d'update pour continuer dans le graph

        Enfin on lance les notification
        """
        metadata = self._get_metadata(input, audio, image)
        config = self.config
        thread_id = 0
        config["configurable"]["thread_id"] = thread_id # change it when the context or user changes
        user_input = {"messages": [HumanMessage(content=metadata.get("text"), id=0)]}

        for namespace, event in self.agent.stream(user_input, config, stream_mode="updates", debug=self.debug, subgraphs=self.subgraphs):
            print(namespace, event)
            ai_res = self.get_ai_answer(config)
            if ai_res:
                print("__ on_submit() Agent is requesting an information __")
                break

        # debug the state
        state = self.get_state(config, show = True)

        # parrot mode (should yield the message in ai_res)
        yield metadata.get("text")
