# Objective
Build a working prototype of a lightweight Agentic AI System that performs end-to-end financial
automation:
Ingests a vendor invoice (PDF or image)
Extracts key fields using an LLM-powered tool
Uses Retrieval-Augmented Generation (RAG) to answer financial queries
Makes a decision on whether to post to QuickBooks
Executes the QuickBooks API call to post the transaction
This demo simulates an Agentic AI workflow, which sees, reasons, and acts on real-world financial data.
# Agent Architecture
The agent should:
Receive an uploaded invoice (PDF or image)
Use tools to parse and embed invoice data
Store data in a vector DB and enable Q&A via RAG
Decide whether to post to QuickBooks based on context or user input
Execute the action through the QuickBooks API
Use LangChain’s AgentExecutor and Tool components to simulate an autonomous reasoning loop.

# Execution Process for Demo

# Key Files
**main.py**: FastAPI app with endpoints (/upload, /ask).
**config.py**: Environment and Pinecone setup.
**utils.py**: QuickBooks API, text extraction, and agent tools.
**demo.html**: Frontend interface

**Note:** To execute this project, make sure to create and configure a .env file with your own credentials. Please find the required .env file fields listed at the end of this file.

This document outlines the process of executing the demo application stored in the `FrontEnd` folder, which interacts with the FastAPI backend for invoice processing and question-answering functionalities.

## Prerequisites
Before running the demo, ensure the following are set up:
- **Python Environment**: Python 3.8+ with dependencies installed (see `requirements.txt` in the root directory).
- **Backend Setup**: The FastAPI backend (`main.py`) must be running on `http://127.0.0.1:8000`.
- **Environment Variables**: A `.env` file with required configurations (e.g., QuickBooks credentials, Google API key, Pinecone API key) as defined in `config.py`.
- **Node.js and npm**: Required for running the frontend application.
- **Supported Browsers**: Modern browsers (Chrome, Firefox, Edge) for viewing the `demo.html` interface.

## Folder Structure
The `FrontEnd` folder contains:
- `demo.html`: The main HTML file providing the user interface for file uploads and question-asking.
- (Optional) Additional static assets (e.g., CSS, JavaScript) if added for styling or interactivity.

## Steps to Execute the Demo
1. **Start the Backend Server**
   - Navigate to the root directory containing `main.py`.
   - Ensure all dependencies are installed:
     ```bash
     pip install -r requirements.txt
     ```
   - Run the FastAPI application:
     ```bash
     uvicorn main:app --host 127.0.0.1 --port 8000
     ```
   - Verify the server is running by accessing `http://127.0.0.1:8000/docs` in a browser, which displays the FastAPI Swagger UI.

2. **Serve the Frontend**
   - Navigate to the `FrontEnd` folder:
     ```bash
     cd FrontEnd
     ```
   - If `demo.html` is a static file, serve it using a simple HTTP server:
     ```bash
     python -m http.server 8080
     ```
   - Alternatively, if the frontend uses a framework (e.g., React), ensure it is built and served:
     ```bash
     npm install
     npm start
     ```
     (Note: Update this step if additional frontend setup is required.)

3. **Access the Demo**
   - Open a browser and navigate to `http://127.0.0.1:8080/demo.html`.
   - The interface displays:
     - A file upload section for PDF or image files (e.g., invoices).
     - A text input for asking questions about uploaded files.
     - A response area to display results (e.g., extracted text, vendor details, posting status).

4. **Upload a File**
   - Use the file input to upload a PDF or image (PNG, JPG, JPEG) invoice.
   - The frontend sends the file to the backend endpoint `/upload` (POST request).
   - The backend:
     - Validates the file type and size.
     - Extracts text using `PyPDF2` (for PDFs) or `EasyOCR` (for images).
     - Stores the extracted text in Pinecone for retrieval.
     - Extracts vendor name and amount using a conversational retrieval chain.
     - Uses a LangChain agent to evaluate and post the bill to QuickBooks if conditions are met (vendor exists, amount < $10,000).
   - The response includes:
     - Filename, status, extracted text, vendor name, amount, posting status, and bill details.

5. **Ask Questions**
   - Enter a question in the text input (e.g., "Who is the vendor?" or "What is the amount?").
   - Optionally specify a filename to filter the context.
   - The frontend sends the question to the `/ask` endpoint (POST request).
   - The backend uses the conversational retrieval chain to answer based on the Pinecone vector store.
   - The response is displayed in the interface.

6. **View Results**
   - File upload results show extracted data and QuickBooks posting status.
   - Question responses are displayed in the response area.

## Troubleshooting
- **Backend Errors**: Check `app.log` for detailed error messages if uploads or questions fail.
- **CORS Issues**: Ensure the frontend URL is allowed in the backend's CORS configuration (update `main.py` for production).
- **Missing Dependencies**: Verify all required packages are installed.
- **API Key Issues**: Ensure valid credentials in the `.env` file for QuickBooks, Google, and Pinecone.

## Notes
- The demo supports files up to 10MB and only PDF, PNG, JPG, or JPEG formats.
- The backend uses Google Gemini (`gemini-1.5-flash`) for LLM tasks and Pinecone for vector storage.
- QuickBooks integration requires a valid sandbox or production account.
- For production, secure the CORS settings and use HTTPS.



GOOGLE_CLOUD_PROJECT= ****************

GOOGLE_CLOUD_LOCATION= 

GOOGLE_APPLICATION_CREDENTIALS= "path"

GOOGLE_API_KEY = *************************************


PINECONE_API_KEY= ******************************************

PINECONE_INDEX=finz-invoices



QB_CLIENT_ID = ***************************************

QB_CLIENT_SECRET = *************************

QB_REFRESH_TOKEN = ********************

QB_REALM_ID = *************
