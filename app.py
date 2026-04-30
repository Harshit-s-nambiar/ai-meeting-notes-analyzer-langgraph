import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict
from dotenv import load_dotenv
import os

load_dotenv()

# ======================
# LLM SETUP (OpenRouter)
# ======================
llm = ChatOpenAI(
    model="openai/gpt-3.5-turbo",   
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0
)

# Define graph state 
from typing import TypedDict, List, Dict
from urllib import response

class MeetingNotesState(TypedDict):
    transcript: str
    topics: List[str]
    summary: str
    action_items: List[Dict[str, str]]
    priority: str
    report: str


# input node
def input_node(state: MeetingNotesState):
    return state

# topic extraction agent

def extract_topics(state: MeetingNotesState):
    prompt = f"""
            Extract key topics.

            Return ONLY short phrases.
            No bullets, no numbering, no explanation.

            Transcript:
            {state['transcript']}
             """

    response = llm.invoke(prompt).content
    topics = [line.strip() for line in response.split("\n") if line.strip()]
    new_state = state.copy()
    new_state["topics"] = topics
    return new_state

# summary agent

def summarize_meeting(state: MeetingNotesState):
    prompt = f"""Summarize the following meeting transcript in 3 to 5 sentences.
    transcript: {state['transcript']}"""

    response = llm.invoke(prompt).content

    new_state = state.copy()
    new_state["summary"] = response
    return new_state

# action item extraction agent

def extract_action_items(state: MeetingNotesState):
    prompt = f"""
Extract ALL action items from the meeting.

Definition:
An action item is ANY task, responsibility, or commitment assigned to a person.

Rules:
- Include ALL tasks (even small ones)
- Do NOT miss any
- Format STRICTLY as: task | owner
- One per line
- Use exact role names from transcript (e.g., Mobile Developer, QA Tester)
- If no owner, write: Unassigned
- Do NOT include headings or explanations

Transcript:
{state['transcript']}
"""

    response = llm.invoke(prompt).content

    action_items = []

    for line in response.split("\n"):
        if "|" in line:
            parts = line.split("|")
            if len(parts) == 2:
                task = parts[0].strip()
                owner = parts[1].strip()
                action_items.append({"task": task, "owner": owner})

    new_state = state.copy()
    new_state["action_items"] = action_items
    return new_state

# priority classification agent
def classify_priority(state: MeetingNotesState):
    prompt = f"""Determine priority: High / Medium / Low
    transcript: {state['transcript']}"""

    response = llm.invoke(prompt).content

    new_state = state.copy()
    new_state["priority"] = response.strip()
    return new_state


# conditional logic

def check_action_items(state: MeetingNotesState):
    if len(state["action_items"]) == 0:
        return "no_action"
    else:
        return "has_action"


# final output node

def final_output(state):

    report = ""

    # Title
    report += "Meeting Summary\n"
    report += state["summary"] + "\n\n"

    # Topics (numbered)
    report += "Key Topics\n"
    for i, t in enumerate(state["topics"], 1):
        report += f"{i}. {t}\n"

    report += "\n"

    # Action Items (numbered)
    report += "Action Items\n"
    for i, item in enumerate(state["action_items"], 1):
        report += f"{i}. {item['task']} – {item['owner']}\n"

    report += "\n"

    # Priority
    report += "Priority\n"
    report += state["priority"]

    return {"report": report}




# Build the graph

builder = StateGraph(MeetingNotesState)
builder.add_node("input", input_node)
builder.add_node("extract_topics", extract_topics)
builder.add_node("summarize_meeting", summarize_meeting)
builder.add_node("extract_action_items", extract_action_items)
builder.add_node("classify_priority", classify_priority)
builder.add_node("final_output", final_output)

builder.set_entry_point("input")

builder.add_edge("input", "extract_topics")
builder.add_edge("extract_topics", "summarize_meeting")
builder.add_edge("summarize_meeting", "extract_action_items")
builder.add_conditional_edges("extract_action_items", check_action_items,
                             {"has_action": "classify_priority",
                               "no_action": "final_output"}
                             )
builder.add_edge("classify_priority", "final_output")
builder.add_edge("final_output", END)

graph = builder.compile()




# STREAMLIT UI

st.set_page_config(page_title="AI Meeting Notes Analyzer", layout="wide")

st.title(" AI Meeting Notes Analyzer")
st.markdown("Upload or paste a meeting transcript to generate structured notes.")

# Input
transcript_input = st.text_area(" Paste Meeting Transcript", height=300)

uploaded_file = st.file_uploader("Or upload .txt file", type=["txt"])

if uploaded_file:
    transcript_input = uploaded_file.read().decode("utf-8")

# Run button
if st.button("Analyze Meeting"):

    if not transcript_input.strip():
        st.warning("Please provide a transcript.")
    else:
        with st.spinner("Analyzing..."):

            result = graph.invoke({
                "transcript": transcript_input,
                "topics": [],
                "summary": "",
                "action_items": [],
                "priority": ""
            })

        st.success("Analysis Complete!")

        # ======================
        # OUTPUT DISPLAY
        # ======================

        st.subheader(" Summary")
        st.write(result["summary"])

        st.subheader(" Key Topics")
        for i, topic in enumerate(result["topics"], 1):
            st.write(f"{i}. {topic}")

        st.subheader("Action Items")
        if result["action_items"]:
            st.table(result["action_items"])
        else:
            st.write("No action items found")

        st.subheader("Priority")
        st.markdown(f"**{result['priority']}**")