import os
import requests
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
import shutil
import uvicorn
from PyPDF2 import PdfReader
from PIL import Image
import easyocr
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
# from langchain.tools import tool
from langchain.agents import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
import json
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
# from langchain.agents import render_text_description
from langchain.schema import AgentFinish

load_dotenv()

# QuickBooks Environment Variables
QB_CLIENT_ID = os.getenv("QB_CLIENT_ID")
QB_CLIENT_SECRET = os.getenv("QB_CLIENT_SECRET")
QB_REFRESH_TOKEN = os.getenv("QB_REFRESH_TOKEN")
QB_REALM_ID = os.getenv("QB_REALM_ID")
QB_BASE_URL = "https://sandbox-quickbooks.api.intuit.com"

# Google Cloud and Pinecone Settings
PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "langchain-pinecone-rag")

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embeddings = VertexAIEmbeddings(model_name="text-embedding-004")
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = PINECONE_INDEX

# For production, remove the delete logic to persist the index
# if pc.has_index(index_name):
#     pc.delete_index(index_name)

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(index_name)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)
llm = VertexAI(model_name="gemini-2.5-pro", temperature=0.2)


retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# memory.clear() 
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

def get_access_token():
    """Refresh and get a new access token using the refresh token."""
    url = "https://oauth.platform.intuit.com/oauth2/v1/tokens/bearer"
    payload = {
        "grant_type": "refresh_token",
        "refresh_token": QB_REFRESH_TOKEN
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    auth = (QB_CLIENT_ID, QB_CLIENT_SECRET)

    response = requests.post(url, data=payload, auth=auth, headers=headers)
    response.raise_for_status()
    tokens = response.json()
    print("✅ Access token obtained successfully.")
    return tokens["access_token"]

def find_vendor(access_token, vendor_name):
    """Check if a vendor exists by name."""
    url = f"{QB_BASE_URL}/v3/company/{QB_REALM_ID}/query?minorversion=65"
    query = f"select * from Vendor where DisplayName = '{vendor_name}'"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json"
    }
    response = requests.get(url, headers=headers, params={"query": query})
    response.raise_for_status()
    data = response.json()

    vendors = data.get("QueryResponse", {}).get("Vendor", [])
    if vendors:
        vendor_id = vendors[0]["Id"]
        print(f"✅ Vendor '{vendor_name}' found with ID: {vendor_id}")
        return vendor_id
    else:
        print(f"❌ Vendor '{vendor_name}' not found.")
        return None

def create_bill(access_token, vendor_id, amount):
    """Create a bill for the given vendor ID."""
    url = f"{QB_BASE_URL}/v3/company/{QB_REALM_ID}/bill?minorversion=65"
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
                "AccountBasedExpenseLineDetail": {
                    "AccountRef": {"value": "7"}  # Replace with valid account ID
                }
            }
        ],
        "TxnDate": datetime.now().strftime("%Y-%m-%d"),
        "DocNumber": f"BILL-{datetime.now().strftime('%y%m%d%H%M%S')}",
        "DueDate": (datetime.now().replace(day=28).strftime("%Y-%m-%d")),
        "CurrencyRef": {"value": "USD"}
    }

    response = requests.post(url, headers=headers, json=bill_data)
    if response.status_code != 200:
        print("❌ Error creating bill:", response.json())
        return None
    else:
        print("✅ Bill created successfully:", response.json())
        return response.json()

def store_in_pinecone(filename, extracted_text: str):
    try:
        global vector_store
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_text(extracted_text)

        docs = [
            Document(
                page_content=chunk,
                metadata={"filename": filename}
            )
            for chunk in splits
        ]

        vector_store.add_documents(docs)
        print(f"Stored {len(splits)} chunks for file '{filename}' in Pinecone.")
    except Exception as e:
        print(f"Error storing in Pinecone: {str(e)}")
        raise

def extract_text_from_pdf(file_path):
    """Extract text from PDF file"""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    print(f"Extracted text from PDF: {text[:100]}...")
    return text.strip()

def extract_text_from_image(file_path):
    """Extract text from image using EasyOCR"""
    reader = easyocr.Reader(['en'])
    results = reader.readtext(file_path, detail=0)
    extracted_text = " ".join(results)
    print(f"Extracted text from image: {extracted_text}")
    return extracted_text.strip()


def extract_invoice_data(file_filename: str) -> dict:
    """Extract vendor name and amount from the invoice using conversational retrieval chain."""
    print(f"Extracting data from invoice: {file_filename}")
    global qa_chain
    vendor_response = qa_chain.run("Who is the Vendor? Provide just the name. If not found, return 'Unknown Vendor'. search for Vendor")
    amount_response = qa_chain.run(" what is the total amount due? Provide just the numerical value without currency symbols.")

    vendor_name = vendor_response.strip()
    try:
        amount = float(amount_response.strip().replace(",", ""))
    except ValueError:
        amount = None
    print(f"Extracted vendor: {vendor_name}, amount: {amount}")
    return {
        "vendor_name": vendor_name,
        "amount": amount
    }

# def query_invoice_chain(file_filename: str, question: str) -> str:
#     """
#     Ask a question about a previously uploaded invoice using the conversational retrieval chain.
    
#     file_filename: name of the uploaded invoice (must be already stored in Pinecone)
#     question: the question you want to ask about the invoice
#     """
#     # Optionally, you can set up a retriever filter if needed
#     global qa_chain
#     response = qa_chain.run(question)
#     print(f"QA Chain response for '{question}': {response}")
#     return response

def check_posting_conditions(vendor_name: str, amount: float) -> dict:
    """Check if the vendor exists in QuickBooks and if the amount is less than $10,000."""
    if not vendor_name or amount is None:
        return {"can_post": False, "reason": "Failed to extract vendor or amount", "vendor_id": None, "access_token": None}

    access_token = get_access_token()
    vendor_id = find_vendor(access_token, vendor_name)

    if vendor_id and amount < 10000:
        return {"can_post": True, "reason": "Conditions met", "vendor_id": vendor_id, "access_token": access_token}
    else:
        return {"can_post": False, "reason": "Vendor not recognized or amount >= $10,000", "vendor_id": vendor_id, "access_token": access_token}



def decide_to_post(conditions: dict) -> dict:
    """Use an LLM prompt to decide whether to post the bill based on the conditions."""
    prompt_template = PromptTemplate(
        input_variables=["conditions"],
        template="""Based on the following conditions for posting a bill to QuickBooks:

Conditions: {conditions}

Decide whether to auto-post the bill. Respond with 'YES' if you decide to post, 'NO' otherwise, and provide a brief reason.

Decision:"""
    )

    prompt = prompt_template.format(conditions=json.dumps(conditions))
    decision_response = llm.invoke(prompt)

    lines = decision_response.strip().split("\n")
    decision = lines[0].strip().upper() == "YES"
    reason = lines[1].strip() if len(lines) > 1 else "No reason provided"

    return {
        "decision": decision,
        "reason": reason
    }



def post_bill_if_approved(decision: dict, vendor_id: str, amount: float, access_token: str) -> dict:
    """Post the bill to QuickBooks if the decision is to post."""
    if decision.get("decision", False) and vendor_id and access_token:
        bill_details = create_bill(access_token, vendor_id, amount)
        if bill_details:
            return {"posting_status": "Auto-posted successfully", "bill_details": bill_details}
        else:
            return {"posting_status": "Failed to post bill", "bill_details": None}
    else:
        return {"posting_status": f"Did not auto-post: {decision.get('reason', 'Unknown reason')}", "bill_details": None}


from langchain.agents import Tool


# query_invoice_tool = Tool(
#     name="query_invoice_chain",
#     func=query_invoice_chain,
#     description=(
#         "Use this tool to ask questions about an uploaded invoice. "
#         " your question as 'question'. "
#         "The tool queries the conversational retrieval chain to return an answer from the stored invoice content."
#         "Example questions include:"
#         "Based on the uploaded invoice content, what is the vendor name? Provide just the name."
#         "Based on the uploaded invoice content, what is the total amount due? Provide just the numerical value without currency symbols."

#     )
# )

extract_invoice_data_tool = Tool(
    name="extract_invoice_data",
    func=extract_invoice_data,
    description="This will extract the vendor_name and amount and retun vendor_name and amount. Use these values to the check posting conditions."
)

check_posting_conditions_tool = Tool(
    name="check_posting_conditions",
    func=check_posting_conditions,
    description="Check if vendor exists in QuickBooks and if the amount is less than $10,000."
)

decide_to_post_tool = Tool(
    name="decide_to_post",
    func=decide_to_post,
    description="Use an LLM prompt to decide whether to post the bill based on conditions."
)

post_bill_if_approved_tool = Tool(
    name="post_bill_if_approved",
    func=post_bill_if_approved,
    description="Post the bill to QuickBooks if the decision is to post."
)

tools = [
  
    check_posting_conditions_tool,
    decide_to_post_tool,
    post_bill_if_approved_tool
]

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="proto.marshal.rules.enums")


tool_names = [t.name for t in tools]
print("Available tools:", tool_names)
# tool_desc = render_text_description(tools)
tool_desc = "\n".join([f"{t.name}: {t.description}" for t in tools])

react_prompt = PromptTemplate(
    input_variables=["input", "tool_names", "tools_desc", "agent_scratchpad"],
    template="""Answer the following question step-by-step using the available tools.
        Do not attempt to answer without calling these tools.
        You must use the tools in the order specified below.
        You must use the tools in this sequence:
        
        1. check_posting_conditions
        2. decide_to_post
        3. post_bill_if_approved

        Question: {input}
        from input get 'vendor_name' and 'amount' and perform the following instructions in order:
        step-by-step instructions:
           1. check_posting_conditions take input of vendor_name and amount will check if the vendor exists in QuickBooks and if the amount is less than $10,000.
            conditions  =  check_posting_conditions("vendor_name": vendor_name, "amount": amount)
            
           2. decide_to_post will use an LLM prompt to decide whether to post the bill based on conditions.
           decision = decide_to_post(conditions)
           example of conditions:
           {"can_post": True, "reason": "Conditions met", "vendor_id": vendor_id, "access_token": access_token}
           3. post_bill_if_approved will post the bill to QuickBooks if the decision is to post
           posting_result = post_bill_if_approved(decision: dict, vendor_id: str, amount: float, access_token: str) 
           4. You must use the tools in this sequence.
        strictly follow the order of tools specified above.

        Tools available: {tool_names}
        {tools_desc}
        {tools}
        {agent_scratchpad}
        Excute the tools in the order specified above. then make sure to return the final result in strict JSON format:
        Also explain the steps you took to arrive at the final result.
        Make veriable to indeicate you have posted the bill to QuickBooks and execute the tools in the order specified above.
        Add that to the final result.
        After executing each tool print the output of the tool.
        Strickly print the output of each tool. 
        Do not do any other actions other than executing the tools in the order specified above.
        Make sure to return the final result in strict JSON format:
        Provide the final result in strict JSON format:
        {{
        "vendor_name": str,
        "amount": float,
        "posting_status": str,
        "bill_details": dict or null
        
        }}"""
    )

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=10,
    return_intermediate_steps=True
)



# @app.post("/upload")
# async def upload_file(file: UploadFile = File(...)):
#     file_path = os.path.join(UPLOAD_DIR, file.filename)
#     with open(file_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)

#     extracted_text = ""
#     file_ext = file.filename.lower()
#     posting_status = "Not attempted"
#     bill_details = None
#     tool_results = []

#     try:
#         if file_ext.endswith(".pdf"):
#             extracted_text = extract_text_from_pdf(file_path)
#         elif file_ext.endswith((".png", ".jpg", ".jpeg")):
#             extracted_text = extract_text_from_image(file_path)
#         else:
#             extracted_text = "Unsupported file type for text extraction."
#             return {
#                 "filename": file.filename,
#                 "status": "success",
#                 "extracted_text": extracted_text,
#                 "posting_status": "Unsupported file type"
#             }

#         # Store in Pinecone
#         print("extracted_text:", extracted_text)  # Print first 100 characters
#         store_in_pinecone(file.filename, extracted_text)

#         # Set up file-specific retriever with filter
#         global agent_executor, tool_names, tool_desc
#         # Run the agent to process the invoice
#         k = extract_invoice_data(file.filename)
#         # agent_input = f"Take the vendor name and amout as {k} check conditions, decide to post, and post if approved."
#         agent_input = f"vendor_name: {k['vendor_name']}, amount: {k['amount']}"
#         print("running agent with input:", agent_input)
#         response = agent_executor.invoke({
#             "input": agent_input,
#             "tool_names": tool_names,
#             "tools_desc": tool_desc
#         })
#         print("Agent response:", response)
#         intermediate_steps = response["intermediate_steps"][-1][-1]
#         # for step in intermediate_steps:
#         #     action, output = step
#         #     tool_results.append({
#         #         "tool_name": action.tool,
#         #         "tool_input": action.tool_input,
#         #         "tool_output": output
#         #     })
#         print("Intermediate steps:", intermediate_steps)
#         # Parse JSON output safely
#         try:
#             final_result = json.loads(response["output"])
#         except json.JSONDecodeError:
#             raise ValueError(f"Invalid JSON output: {response['output']}")

#         print(final_result)
#         # Parse the agent's final output
#         try:
#             # Assuming the agent returns a JSON string in the required format
#             result = eval(response["output"]) if isinstance(response["output"], str) else response["output"]
#             vendor_name = result.get("vendor_name")
#             amount = result.get("amount")
#             posting_status = result.get("posting_status")
#             bill_details = result.get("bill_details")
#         except Exception as e:
#             print(f"Error parsing agent output: {str(e)}")
#             posting_status = "Failed to parse agent response"
#             bill_details = None

#     except Exception as e:
#         extracted_text = f"Error extracting text or processing: {str(e)}"
#         posting_status = f"Error: {str(e)}"

#     return {
#         "filename": file.filename,
#         "status": "success",
#         "extracted_text": extracted_text,
#         "vendor_extracted": vendor_name if 'vendor_name' in locals() else None,
#         "amount_extracted": amount if 'amount' in locals() else None,
#         "posting_status": posting_status,
#         "bill_details": bill_details
#     }


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    extracted_text = ""
    posting_status = "Not attempted"
    bill_details = None
    vendor_name = None
    amount = None

    try:
        # Extract text
        file_ext = file.filename.lower()
        if file_ext.endswith(".pdf"):
            extracted_text = extract_text_from_pdf(file_path)
        elif file_ext.endswith((".png", ".jpg", ".jpeg")):
            extracted_text = extract_text_from_image(file_path)
        else:
            return {
                "filename": file.filename,
                "status": "error",
                "error": "Unsupported file type"
            }

        print(f"Extracted text: {extracted_text[:120]}...")
        store_in_pinecone(file.filename, extracted_text)

        # Extract vendor and amount
        k = extract_invoice_data(file.filename)
        vendor_name = k["vendor_name"]
        amount = k["amount"]

        agent_input = f"vendor_name: {vendor_name}, amount: {amount}"
        print("running agent with input:", agent_input)
        global agent_executor, tool_names, tool_desc
        response = agent_executor.invoke({
            "input": agent_input,
            "tool_names": tool_names,
            "tools_desc": tool_desc
        })

        # Debug: Print agent response and intermediate steps
        print("Agent response:", response)
        print("Agent output field:", response.get("output"))
        print("Intermediate steps:", response.get("intermediate_steps"))

        # Parse agent output
        output = response.get("output")
        if not output:
            posting_status = "No output from agent"
            bill_details = None
        else:
            try:
                final_result = json.loads(output)
                posting_status = final_result.get("posting_status")
                bill_details = final_result.get("bill_details")
            except Exception as e:
                posting_status = f"Error parsing agent output: {e}"
                bill_details = None

    except Exception as e:
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
    uvicorn.run(app, host="127.0.0.1", port=8000)
