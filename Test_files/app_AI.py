import os
import json
import logging
import requests
from datetime import datetime
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
from PyPDF2 import PdfReader
from PIL import Image
import easyocr
from langchain_google_vertexai import VertexAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, SystemMessage
from langchain.agents import Tool, AgentExecutor, create_tool_calling_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import warnings
import getpass
from pydantic import BaseModel, Field
from typing import Optional, Dict

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="proto.marshal.rules.enums")

# Configuration management
class Settings(BaseSettings):
    QB_CLIENT_ID: str
    QB_CLIENT_SECRET: str
    QB_REFRESH_TOKEN: str
    QB_REALM_ID: str
    QB_BASE_URL: str = "https://sandbox-quickbooks.api.intuit.com"
    GOOGLE_CLOUD_PROJECT: str
    GOOGLE_CLOUD_LOCATION: str = "us-central1"
    GOOGLE_API_KEY: str
    PINECONE_API_KEY: str
    PINECONE_INDEX: str = "langchain-pinecone-rag"
    UPLOAD_DIR: str = "Uploads"
    GOOGLE_APPLICATION_CREDENTIALS: str
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Load environment variables and validate
load_dotenv()
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")
settings = Settings()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("app.log")]
)
logger = logging.getLogger(__name__)


# Custom exceptions
class QuickBooksError(Exception):
    pass

class TextExtractionError(Exception):
    pass

class AgentExecutionError(Exception):
    pass

# Create upload directory
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

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
pc = Pinecone(api_key=settings.PINECONE_API_KEY)

index_name = settings.PINECONE_INDEX

# if pc.has_index(index_name):
#     pc.delete_index(index_name)
# Create Pinecone index if it doesn't exist
if index_name not in [index['name'] for index in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)
llm = init_chat_model("gemini-1.5-flash", model_provider="google_genai", temperature=0.2)

# Initialize retriever and conversation chain
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3, "filter": {"filename": {"$exists": True}}})
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

# QuickBooks API functions
def get_access_token() -> str:
    """Refresh and get a new access token using the refresh token."""
    try:
        url = "https://oauth.platform.intuit.com/oauth2/v1/tokens/bearer"
        payload = {"grant_type": "refresh_token", "refresh_token": settings.QB_REFRESH_TOKEN}
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        auth = (settings.QB_CLIENT_ID, settings.QB_CLIENT_SECRET)
        response = requests.post(url, data=payload, auth=auth, headers=headers)
        response.raise_for_status()
        tokens = response.json()
        logger.info("Access token obtained successfully")
        return tokens["access_token"]
    except Exception as e:
        logger.error(f"Failed to get access token: {str(e)}")
        raise QuickBooksError(f"Failed to get access token: {str(e)}")

def find_vendor(access_token: str, vendor_name: str) -> str | None:
    """Check if a vendor exists by name."""
    try:
        url = f"{settings.QB_BASE_URL}/v3/company/{settings.QB_REALM_ID}/query?minorversion=65"
        query = f"select * from Vendor where DisplayName = '{vendor_name}'"
        headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}
        response = requests.get(url, headers=headers, params={"query": query})
        response.raise_for_status()
        data = response.json()
        vendors = data.get("QueryResponse", {}).get("Vendor", [])
        if vendors:
            vendor_id = vendors[0]["Id"]
            logger.info(f"Vendor '{vendor_name}' found with ID: {vendor_id}")
            return vendor_id
        logger.warning(f"Vendor '{vendor_name}' not found")
        return None
    except Exception as e:
        logger.error(f"Failed to find vendor '{vendor_name}': {str(e)}")
        raise QuickBooksError(f"Failed to find vendor: {str(e)}")

def create_bill(access_token: str, vendor_id: str, amount: float) -> dict | None:
    """Create a bill for the given vendor ID."""
    try:
        url = f"{settings.QB_BASE_URL}/v3/company/{settings.QB_REALM_ID}/bill?minorversion=65"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        bill_data = {
            "VendorRef": {"value": vendor_id},
            "Line": [
                {
                    "Amount": amount,
                    "DetailType": "AccountBasedExpenseLineDetail",
                    "AccountBasedExpenseLineDetail": {"AccountRef": {"value": "7"}}  # Replace with valid account ID
                }
            ],
            "TxnDate": datetime.now().strftime("%Y-%m-%d"),
            "DocNumber": f"BILL-{datetime.now().strftime('%y%m%d%H%M%S')}",
            "DueDate": (datetime.now().replace(day=28).strftime("%Y-%m-%d")),
            "CurrencyRef": {"value": "USD"}
        }
        response = requests.post(url, headers=headers, json=bill_data)
        response.raise_for_status()
        logger.info("Bill created successfully")
        return response.json()
    except Exception as e:
        logger.error(f"Failed to create bill: {str(e)}")
        return None

# File processing functions
def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text
        logger.info(f"Extracted text from PDF {file_path}: {text[:100]}...")
        return text.strip()
    except Exception as e:
        logger.error(f"Failed to extract text from PDF {file_path}: {str(e)}")
        raise TextExtractionError(f"Failed to extract text from PDF: {str(e)}")

def extract_text_from_image(file_path: str) -> str:
    """Extract text from an image using EasyOCR."""
    try:
        reader = easyocr.Reader(['en'])
        results = reader.readtext(file_path, detail=0)
        text = " ".join(results)
        logger.info(f"Extracted text from image {file_path}: {text[:100]}...")
        return text.strip()
    except Exception as e:
        logger.error(f"Failed to extract text from image {file_path}: {str(e)}")
        raise TextExtractionError(f"Failed to extract text from image: {str(e)}")

def store_in_pinecone(filename: str, extracted_text: str):
    """Store extracted text in Pinecone."""
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_text(extracted_text)
        docs = [Document(page_content=chunk, metadata={"filename": filename}) for chunk in splits]
        vector_store.add_documents(docs)
        logger.info(f"Stored {len(splits)} chunks for file '{filename}' in Pinecone")
    except Exception as e:
        logger.error(f"Failed to store in Pinecone for {filename}: {str(e)}")
        raise

# Invoice data extraction
def extract_invoice_data(file_filename: str) -> dict:
    """Extract vendor name and amount from the invoice using conversational retrieval chain."""
    try:
        logger.info(f"Extracting data from invoice: {file_filename}")
        
        vendor_response = qa_chain.run("Who is the Vendor? Provide just the name. If not found, return 'Unknown Vendor'.")
        amount_response = qa_chain.run("What is the total amount due? Provide just the numerical value without currency symbols.")
        vendor_name = vendor_response.strip()
        print(amount_response)
        try:
            amount_response = amount_response.replace("$", "").replace("USD", "").strip()
            amount = float(amount_response.strip().replace(",", ""))
        except ValueError:
            amount = None
        logger.info(f"Extracted vendor: {vendor_name}, amount: {amount}")
        return {"vendor_name": vendor_name, "amount": amount}
    except Exception as e:
        logger.error(f"Failed to extract invoice data for {file_filename}: {str(e)}")
        raise AgentExecutionError(f"Failed to extract invoice data: {str(e)}")

# Agent tools
class DecisionDict(BaseModel):
    decision: bool = Field(description="Decision to post the bill")
    reason: str = Field(description="Reason for the decision")
    vendor_id: Optional[str] = Field(description="Vendor ID if found")
    access_token: Optional[str] = Field(description="Access token for QuickBooks")

class EvaluateInput(BaseModel):
    vendor_name: str = Field(description="Name of the vendor")
    amount: float = Field(description="Invoice amount in USD")

class PostBillInput(BaseModel):
    decision_dict: DecisionDict = Field(description="Decision dictionary from evaluation")
    amount: float = Field(description="Invoice amount in USD")

def evaluate_and_decide_posting(vendor_name: str, amount: float) -> Dict:
    """
    Check posting conditions and decide whether to post a bill to QuickBooks.
    Returns a dictionary with decision (bool), reason (str), vendor_id (str or None), and access_token (str or None).
    """
    try:
        if not vendor_name or amount is None:
            logger.warning("Invalid vendor or amount")
            return {
                "decision": False,
                "reason": "Failed to extract vendor or amount",
                "vendor_id": None,
                "access_token": None
            }

        access_token = get_access_token()
        vendor_id = find_vendor(access_token, vendor_name)
        if not vendor_id or amount >= 10000:
            logger.warning(f"Cannot post: Vendor not found or amount >= $10,000")
            return {
                "decision": False,
                "reason": "Vendor not recognized or amount >= $10,000",
                "vendor_id": vendor_id,
                "access_token": access_token
            }

        # # Use LLM to decide whether to post
        # decision_prompt = f"Based on the vendor 'is {vendor_id}' and amount {amount < 10000}, should we auto-post this bill to QuickBooks? Respond with 'Yes' or 'No' followed by a brief reason."
        # response = llm.invoke(decision_prompt)
        # content = response.content.strip().lower()
        # if content.startswith("yes"):
        return {
                "decision": True,
                "reason":  "LLM approved",
                "vendor_id": vendor_id,
                "access_token": access_token
            }
        # else:
        #     return {
        #         "decision": False,
        #         "reason": content.split("no")[1].strip() if len(content.split("no")) > 1 else "LLM declined",
        #         "vendor_id": vendor_id,
        #         "access_token": access_token
        #     }

    except Exception as e:
        logger.error(f"Error evaluating posting conditions: {str(e)}")
        raise AgentExecutionError(f"Error evaluating posting conditions: {str(e)}")

def post_bill_if_approved(decision_dict: DecisionDict, amount: float) -> dict:
    """Post the bill to QuickBooks if approved."""
    try:
        if decision_dict.decision:
            bill_details = create_bill(decision_dict.access_token, decision_dict.vendor_id, amount)
            if bill_details:
                logger.info("Bill posted successfully")
                return {"posting_status": "Auto-posted successfully", "bill_details": bill_details}
            else:
                logger.warning("Failed to create bill")
                return {"posting_status": "Failed to post bill", "bill_details": None}
        else:
            logger.warning(f"Did not auto-post: {decision_dict.reason}")
            return {"posting_status": f"Did not auto-post: {decision_dict.reason}", "bill_details": None}
    except Exception as e:
        logger.error(f"Failed to post bill: {str(e)}")
        raise AgentExecutionError(f"Failed to post bill: {str(e)}")

from langchain.tools import StructuredTool

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
    # memory.clear()  # Clear memory for each new upload
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
            raise AgentExecutionError("Failed to extract valid vendor or amount")

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
                final_result = json.loads(output)
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

    response = qa_chain.run(question)
    return {"answer": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)