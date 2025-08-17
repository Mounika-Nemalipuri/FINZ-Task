import os
import requests
from datetime import datetime
from PyPDF2 import PdfReader
from PIL import Image
import easyocr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_pinecone import PineconeVectorStore
from config import settings
import logging
from pydantic import BaseModel, Field
from typing import Optional, Dict
from langchain_google_vertexai import VertexAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

logger = logging.getLogger(__name__)

# Custom exceptions
class QuickBooksError(Exception):
    pass

class TextExtractionError(Exception):
    pass

class AgentExecutionError(Exception):
    pass

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
        vector_store = PineconeVectorStore(index_name=settings.PINECONE_INDEX, embedding=VertexAIEmbeddings(model_name="text-embedding-004"))
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
        vector_store = PineconeVectorStore(index_name=settings.PINECONE_INDEX, embedding=VertexAIEmbeddings(model_name="text-embedding-004"))
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3, "filter": {"filename": {"$exists": True}}})
        llm = init_chat_model("gemini-1.5-flash", model_provider="google_genai", temperature=0.2)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory
        )
        vendor_response = qa_chain.run("Who is the Vendor? Provide just the name. If not found, return 'Unknown Vendor'.")
        amount_response = qa_chain.run("What is the total amount due? Provide just the numerical value without currency symbols.")
        vendor_name = vendor_response.strip()
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

# Agent tools models
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

        return {
            "decision": True,
            "reason": "LLM approved",
            "vendor_id": vendor_id,
            "access_token": access_token
        }

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