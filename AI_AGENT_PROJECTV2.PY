!pip install langgraph langchain-openai langchain-core streamlit openai requests pydantic python-dotenv
import streamlit as st
import json
import requests
from dotenv import load_dotenv
import os
from pydantic import BaseModel
import openai
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import AzureChatOpenAI

# Load environment variables
load_dotenv()

# Configuration
SERPER_API_KEY = st.secrets.get("SERPER_API_KEY", os.getenv("SERPER_API_KEY"))

# Azure OpenAI client setup
client = openai.AzureOpenAI(
    api_key= st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
,  
    api_version="2023-12-01-preview",
    azure_endpoint="https://openai-api-management-gw.azure-api.net/"
)

# LangChain Azure OpenAI setup for LangGraph
llm = AzureChatOpenAI(
    api_key=st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
,
    api_version="2023-12-01-preview",
    azure_endpoint="https://openai-api-management-gw.azure-api.net/",
    azure_deployment="gpt-4o-mini",
    temperature=0.3
)

# Pydantic models for structured data
class WebSearchPrompt(BaseModel):
    search_query: str
    justification: str

class SearchResult(BaseModel):
    title: str
    link: str
    snippet: str

class FinalAnswer(BaseModel):
    optimized_query: str
    search_results: List[SearchResult]
    summary: str
    justification: str

# State definition for LangGraph
class SearchState(TypedDict):
    user_query: str
    optimized_query: str
    justification: str
    search_results: List[dict]
    final_answer: str
    messages: List

# Step 1: Query Optimization Node
def optimize_query_node(state: SearchState) -> SearchState:
    """Node to optimize the user's query for better web search results."""
    user_query = state["user_query"]
    
    prompt = f"""
    You are an expert at optimizing search queries for web search engines.
    
    Original user question: "{user_query}"
    
    Your task is to:
    1. Create an optimized search query that will yield the most relevant web search results
    2. Provide clear justification for your optimization choices
    
    Consider:
    - Using specific keywords
    - Removing unnecessary words
    - Adding relevant context terms
    - Making it search-engine friendly
    
    Provide your response in this format:
    OPTIMIZED_QUERY: [your optimized query]
    JUSTIFICATION: [explain why this query will work better]
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    
    content = response.choices[0].message.content
    
    # Parse the response
    lines = content.split('\n')
    optimized_query = ""
    justification = ""
    
    for line in lines:
        if line.startswith("OPTIMIZED_QUERY:"):
            optimized_query = line.replace("OPTIMIZED_QUERY:", "").strip()
        elif line.startswith("JUSTIFICATION:"):
            justification = line.replace("JUSTIFICATION:", "").strip()
    
    # Fallback if parsing fails
    if not optimized_query:
        optimized_query = user_query
    if not justification:
        justification = "Using original query as optimization failed."
    
    state["optimized_query"] = optimized_query
    state["justification"] = justification
    state["messages"].append(AIMessage(content=f"Optimized query: {optimized_query}"))
    
    return state

# Step 2: Web Search Node
def search_web_node(state: SearchState) -> SearchState:
    """Node to perform web search using the optimized query."""
    query = state["optimized_query"]
    
    def search_internet(query: str) -> List[dict]:
        """Search the internet using Serper API and return structured results."""
        top_result_to_return = 4
        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": query})
        headers = {
            'X-API-KEY': SERPER_API_KEY,
            'content-type': 'application/json'
        }
        
        try:
            response = requests.post(url, headers=headers, data=payload)
            
            if 'organic' not in response.json():
                return []
            else:
                results = response.json()['organic']
                formatted_results = []
                
                for result in results[:top_result_to_return]:
                    try:
                        formatted_result = {
                            "title": result['title'],
                            "link": result['link'],
                            "snippet": result['snippet']
                        }
                        formatted_results.append(formatted_result)
                    except KeyError:
                        continue
                
                return formatted_results
        
        except Exception as e:
            return [{"title": "Search Error", "link": "", "snippet": f"Search failed: {str(e)}"}]
    
    search_results = search_internet(query)
    state["search_results"] = search_results
    state["messages"].append(AIMessage(content=f"Found {len(search_results)} search results"))
    
    return state

# Step 3: Summarization Node
def summarize_results_node(state: SearchState) -> SearchState:
    """Node to create a comprehensive summary of the search results."""
    user_query = state["user_query"]
    optimized_query = state["optimized_query"]
    search_results = state["search_results"]
    
    # Prepare search results for the prompt
    results_text = ""
    for i, result in enumerate(search_results, 1):
        results_text += f"{i}. **{result['title']}**\n   {result['snippet']}\n   Link: {result['link']}\n\n"
    
    prompt = f"""
    Based on the following search results, provide a comprehensive answer to the user's original question.
    
    Original Question: "{user_query}"
    Search Query Used: "{optimized_query}"
    
    Search Results:
    {results_text}
    
    Please provide:
    1. A direct answer to the user's question based on the search results
    2. Key insights from the search results
    3. Any limitations or additional context needed
    
    Be comprehensive but concise.
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    
    summary = response.choices[0].message.content
    state["final_answer"] = summary
    state["messages"].append(AIMessage(content="Generated comprehensive summary"))
    
    return state

# Create the LangGraph workflow
def create_search_workflow():
    """Create and return the LangGraph workflow."""
    workflow = StateGraph(SearchState)
    
    # Add nodes
    workflow.add_node("optimize_query", optimize_query_node)
    workflow.add_node("search_web", search_web_node)
    workflow.add_node("summarize_results", summarize_results_node)
    
    # Define the workflow edges
    workflow.add_edge("optimize_query", "search_web")
    workflow.add_edge("search_web", "summarize_results")
    workflow.add_edge("summarize_results", END)
    
    # Set entry point
    workflow.set_entry_point("optimize_query")
    
    # Compile the workflow
    return workflow.compile()

# Streamlit UI
def main():
    st.title("🤖 LangGraph-Powered Intelligent Search")
    st.markdown("Enter a question to get an AI-optimized search with comprehensive results and analysis.")
    
    # Add styling
    st.markdown("""
    <style>
    .search-box {
        padding: 10px;
        border-radius: 10px;
        background-color: #f0f2f6;
    }
    .result-card {
        padding: 15px;
        border-radius: 10px;
        background-color: #ffffff;
        border-left: 5px solid #4CAF50;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create the workflow
    app = create_search_workflow()
    
    # Input field
    user_query = st.text_input("🔎 Enter your question:", 
                              placeholder="e.g., What are the latest developments in AI?")
    
    if user_query and st.button("🚀 Search & Analyze", type="primary"):
        with st.spinner("🔄 Processing your query through the AI workflow..."):
            # Initialize state
            initial_state = {
                "user_query": user_query,
                "optimized_query": "",
                "justification": "",
                "search_results": [],
                "final_answer": "",
                "messages": [HumanMessage(content=user_query)]
            }
            
            # Run the workflow
            final_state = app.invoke(initial_state)
        
        # Display results in organized sections
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🎯 Query Optimization")
            st.markdown(f"**Original Query:** {user_query}")
            st.markdown(f"**Optimized Query:** {final_state['optimized_query']}")
            st.markdown(f"**Justification:** {final_state['justification']}")
        
        with col2:
            st.subheader("📊 Search Results Overview")
            st.markdown(f"**Results Found:** {len(final_state['search_results'])}")
            st.markdown(f"**Search Status:** ✅ Complete")
        
        # Display comprehensive answer
        st.subheader("🧠 AI Analysis & Summary")
        st.markdown(final_state["final_answer"])
        
        # Display individual search results
        st.subheader("🔍 Detailed Search Results")
        for i, result in enumerate(final_state["search_results"], 1):
            with st.expander(f"Result {i}: {result['title']}", expanded=False):
                st.markdown(f"**📝 Summary:** {result['snippet']}")
                st.markdown(f"**🔗 Link:** [{result['title']}]({result['link']})")
    
    # Sidebar information
    with st.sidebar:
        st.header("LangGraph Workflow")
        st.write("This application uses LangGraph to orchestrate:")
        st.write("1. **Query Optimization** - AI improves your search query")
        st.write("2. **Web Search** - Searches using optimized query")
        st.write("3. **Result Analysis** - AI analyzes and summarizes findings")
        
                
        
if __name__ == "__main__":
    main()