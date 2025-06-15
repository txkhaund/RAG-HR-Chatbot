from dotenv import load_dotenv
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain.agents import Tool, initialize_agent, AgentType
# from langchain.memory import ConversationBufferMemory

import rag_agent
from rag_agent import rag_answer_tool, calculator_tool, wiki_search_tool

load_dotenv()

OPENAI_MODEL_NAME = "deepseek/deepseek-r1-0528-qwen3-8b:free"
AGENT = None

# Pydantic models for request/response
class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    This block runs exactly once when the app starts up.
    """
    rag_agent.main()
    tool_calc = Tool(
        name="Calculator",
        func=calculator_tool,
        description=(
            "Use this tool to perform simple arithmetic calculations. "
            "Input: a math expression like '15 * 4' or '10 + 23'. Output: the numeric result."
        ),
    )

    tool_wiki = Tool(
        name="Wikipedia",
        func=wiki_search_tool,
        description=(
            "Use this tool to look up general-purpose facts from Wikipedia. "
            "Input: a search query string. Output: the Wikipedia summary."
        ),
    )

    tool_rag_answer = Tool(
        name="RAG_Answer",
        func=rag_answer_tool,
        description=(
            "Use this tool to answer questions about Walmart's policies. "
            "It retrieves relevant chunks, summarize them in 100 words with citations, "
            "and then answer the original question with citations metadata (filename, page, chunk_id). "
            "Input: a user question. Output: a cited, coherant answer or 'Insufficient information...'."
        ),
    )

    tools = [tool_calc, tool_wiki, tool_rag_answer]

    print("Lifespan startup: Initializing Agent with Calculator + Wikipedia + RAG_Answer tools...")
    llm = ChatOpenAI(model=OPENAI_MODEL_NAME, temperature=0.5)

    global AGENT
    AGENT = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,    # Instructs the Agent to use the “ReAct” schema (Reasoning + Action) without needing in-prompt examples.
                                                        # It reads each tool’s description to learn when and how to call it.
        verbose=True,                                   # Set to False to hide Thought/Action/Observation traces
        max_iterations=3,
        # prefix=(
        #     "You are an expert assistant. Whenever the user's question mentions a PDF filename or "
        #     "a page number or Walmart policy, always call RAG_Retriever first. "
        #     "Otherwise, if it's purely math, call Calculator. "
        #     "Then concatenate Observations into the final answer."
        # ),
        # memory= ConversationBufferMemory(memory_key="chat_history", return_messages=True),
    )

    print("Lifespan startup: Agent is ready.")
    yield

app = FastAPI(
    title="Agentic RAG Service",
    lifespan=lifespan,
)


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    POST /ask
    Body: { "question": "some question" }
    Returns: { "answer": "the agent's answer" }
    """
    query = request.question.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        answer = AGENT.run(query)
        return AskResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")

