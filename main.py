import json
import os
import shutil
import warnings
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_google_vertexai import VertexAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from config import Settings, settings
from utils import (
    get_access_token, find_vendor, create_bill, extract_text_from_pdf,
    extract_text_from_image, store_in_pinecone, extract_invoice_data,
    evaluate_and_decide_posting, post_bill_if_approved, DecisionDict, EvaluateInput, PostBillInput
)
from langchain.tools import StructuredTool
import logging
import uvicorn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("app.log")]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# CORS configuration (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AI and vector store
embeddings = VertexAIEmbeddings(model_name="text-embedding-004")
llm = init_chat_model("gemini-1.5-flash", model_provider="google_genai", temperature=0.2)
index_name = settings.PINECONE_INDEX
vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)

# Initialize retriever and conversation chain
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3, "filter": {"filename": {"$exists": True}}})
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

# Agent tools
tools = [
    StructuredTool.from_function(
        name="evaluate_and_decide_posting",
        func=evaluate_and_decide_posting,
        description="Check if vendor exists in QuickBooks, if amount is less than $10,000, and use LLM to decide whether to post the bill. Takes vendor_name (str) and amount (float) as inputs. Returns a decision dictionary.",
        args_schema=EvaluateInput,
        return_direct=False
    ),
    StructuredTool.from_function(
        name="post_bill_if_approved",
        func=post_bill_if_approved,
        description="Post the bill to QuickBooks if approved. Takes decision_dict (DecisionDict) and amount (float) as inputs. Returns a dictionary with posting_status (str) and bill_details (dict or None).",
        args_schema=PostBillInput,
        return_direct=False
    )
]

tool_names = ", ".join([t.name for t in tools])
tool_desc = "\n".join([f"{t.name}: {t.description}" for t in tools])

# Define the prompt for tool-calling agent
system_content = f"""You are an assistant tasked with processing an invoice to determine if a bill should be posted to QuickBooks. You must use the provided tools to complete the task. Follow these steps in order:
You will receive an input that contains the vendor name and amount. Use these values to evaluate posting conditions, decide whether to post, and post the bill if approved.
Do not attempt to answer without calling these tools.
You must use the tools in the following order:
1. Use `evaluate_and_decide_posting` to verify if the vendor exists in QuickBooks, if the amount is less than $10,000, and to decide whether to post.
   Example: result = evaluate_and_decide_posting(vendor_name=vendor_name, amount=amount)
   
2. Use `post_bill_if_approved` to post the bill if the decision is to proceed.
    if result['decision']:
        posting_result = post_bill_if_approved(decision_dict=result, amount=amount)
    else:
        return {{"posting_status": "Not posted", "bill_details": ''}}
   Example: posting_result = post_bill_if_approved(decision_dict=result, amount=amount)
You must use the tools in this sequence.

Available tools: {tool_names}
Tool descriptions:
{tool_desc}

Execute the tools in the specified order and return the final result in strict JSON format:
{{
  "vendor_name": str,
  "amount": float,
  "posting_status": str,
  "bill_details": dict or null
}}
"""

tool_calling_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=system_content),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# Initialize chat history store
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# LangChain agent setup
agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=tool_calling_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=10,
    return_intermediate_steps=False
)

# Wrap agent with chat history
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# FastAPI endpoints
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Handle file upload, extract text, store in Pinecone, and process bill posting."""
    # Validate file type and size
    allowed_extensions = {".pdf", ".png", ".jpg", ".jpeg"}
    file_ext = os.path.splitext(file.filename.lower())[1]
    if file_ext not in allowed_extensions:
        logger.error(f"Unsupported file type: {file_ext}")
        raise HTTPException(status_code=400, detail="Unsupported file type")
    if file.size > 10 * 1024 * 1024:  # 10MB limit
        logger.error("File too large")
        raise HTTPException(status_code=400, detail="File size exceeds 10MB limit")

    file_path = os.path.join(settings.UPLOAD_DIR, file.filename)
    extracted_text = ""
    posting_status = "Not attempted"
    bill_details = None
    vendor_name = None
    amount = None

    try:
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"File saved: {file_path}")

        # Extract text
        if file_ext == ".pdf":
            extracted_text = extract_text_from_pdf(file_path)
        else:
            extracted_text = extract_text_from_image(file_path)
        logger.info(f"Extracted text: {extracted_text[:120]}...")

        # Store in Pinecone
        store_in_pinecone(file.filename, extracted_text)

        # Extract invoice data
        k = extract_invoice_data(file.filename)
        vendor_name = k["vendor_name"]
        amount = k["amount"]
        print(f"Extracted vendor: {vendor_name}, amount: {amount}")
        if not vendor_name or amount is None:
            logger.warning("Failed to extract valid vendor or amount")
            raise HTTPException(status_code=400, detail="Failed to extract valid vendor or amount")

        # Run agent with unique session_id based on filename
        search_kwargs = {"k": 3}
        if file.filename:
            search_kwargs["filter"] = {"filename": file.filename}
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs=search_kwargs)
        global qa_chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory
        )
        session_id = f"upload_{file.filename}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        agent_input = f"Process the invoice with vendor_name: {vendor_name}, amount: {amount}. Do not change the vendor_name or amount. Always take vendor_name: {vendor_name} and amount: {amount} as input."
        logger.info(f"Running agent with input: {agent_input}, session_id: {session_id}")
        response = agent_with_chat_history.invoke(
            {"input": agent_input},
            config={"configurable": {"session_id": session_id}}
        )
        logger.info(f"Agent response: {response}")

        # Parse agent output
        output = response.get("output")
        if not output:
            logger.error("No output from agent")
            posting_status = "No output from agent"
        else:
            try:
                final_result = output
                print(f"Final result from agent: {final_result}")
                
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON output from agent: {str(e)}")
                posting_status = f"Error parsing agent output: {str(e)}"
                bill_details = None

    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {str(e)}")
        extracted_text = f"Error extracting text or processing: {str(e)}"
        posting_status = f"Error: {str(e)}"

    return {
        "filename": file.filename,
        "status": "success" if posting_status and "error" not in posting_status.lower() else "error",
        "extracted_text": extracted_text,
        "vendor_extracted": vendor_name,
        "amount_extracted": amount,
        "posting_status": posting_status,
        "bill_details": bill_details
    }

@app.post("/ask")
async def ask_question(request: Request):
    data = await request.json()
    question = data.get("question", "")
    filename = data.get("filename", None)  # Optional: filter by filename

    search_kwargs = {"k": 3}
    if filename:
        search_kwargs["filter"] = {"filename": filename}
    global qa_chain
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs=search_kwargs)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )
    response = qa_chain.run(question)
    return {"answer": response}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)