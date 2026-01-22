"""
FastAPI Application for BioMed-RAG
Web interface with model selection and question answering
UPDATED: Added TinyLlama support
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import uvicorn
import logging
import traceback

from core.rag_pipeline import BiomedRAG
from models import MODEL_INFO

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="BioMed-RAG API",
    description="Biomedical Question Answering with Retrieval-Augmented Generation",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG instance cache
rag_instances = {}


# Pydantic models
class QuestionRequest(BaseModel):
    question: str = Field(..., description="Biomedical question to answer")
    model: str = Field("tinyllama", description="Model to use (tinyllama, phi3, llama, biomistral)")
    use_rag: bool = Field(True, description="Whether to use retrieval")
    n_docs: int = Field(5, description="Number of documents to retrieve", ge=1, le=20)
    temperature: float = Field(0.7, description="Sampling temperature", ge=0.0, le=2.0)
    max_tokens: int = Field(256, description="Maximum tokens to generate", ge=50, le=1024)


class QuestionResponse(BaseModel):
    question: str
    answer: str
    model: str
    use_rag: bool
    retrieval_time: float
    generation_time: float
    total_time: float
    num_docs_retrieved: int
    retrieved_pmids: List[int]


class HealthResponse(BaseModel):
    status: str
    available_models: List[str]
    faiss_connected: bool


def get_rag_instance(model_name: str, use_rag: bool = True) -> BiomedRAG:
    """Get or create RAG instance for a model."""
    cache_key = f"{model_name}_{use_rag}"
    
    if cache_key not in rag_instances:
        logger.info(f"Creating new RAG instance for {cache_key}")
        try:
            rag_instances[cache_key] = BiomedRAG(
                model_name=model_name,
                n_docs=5,
                rerank=True,
                use_rag=use_rag
            )
            logger.info(f"Successfully created RAG instance for {cache_key}")
        except Exception as e:
            logger.error(f"Failed to create RAG instance: {e}")
            logger.error(traceback.format_exc())
            raise
    
    return rag_instances[cache_key]


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web interface."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>BioMed-RAG</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                max-width: 1000px;
                margin: 0 auto;
                padding: 20px;
                background: #f5f5f5;
            }
            .container {
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #2c3e50;
                margin-bottom: 10px;
            }
            .subtitle {
                color: #7f8c8d;
                margin-bottom: 30px;
            }
            .input-group {
                margin-bottom: 20px;
            }
            label {
                display: block;
                margin-bottom: 5px;
                font-weight: 600;
                color: #34495e;
            }
            input, select, textarea {
                width: 100%;
                padding: 10px;
                border: 2px solid #e0e0e0;
                border-radius: 5px;
                font-size: 14px;
                box-sizing: border-box;
            }
            textarea {
                min-height: 100px;
                font-family: inherit;
                resize: vertical;
            }
            button {
                background: #3498db;
                color: white;
                padding: 12px 30px;
                border: none;
                border-radius: 5px;
                font-size: 16px;
                cursor: pointer;
                font-weight: 600;
            }
            button:hover {
                background: #2980b9;
            }
            button:disabled {
                background: #bdc3c7;
                cursor: not-allowed;
            }
            .answer-box {
                margin-top: 30px;
                padding: 20px;
                background: #ecf0f1;
                border-radius: 5px;
                display: none;
            }
            .answer-box.show {
                display: block;
            }
            .meta-info {
                display: flex;
                gap: 20px;
                margin-top: 15px;
                padding-top: 15px;
                border-top: 2px solid #bdc3c7;
                font-size: 14px;
                color: #7f8c8d;
                flex-wrap: wrap;
            }
            .meta-item {
                display: flex;
                flex-direction: column;
            }
            .meta-label {
                font-weight: 600;
                margin-bottom: 3px;
            }
            .loading {
                text-align: center;
                padding: 20px;
                color: #3498db;
                display: none;
            }
            .error {
                background: #e74c3c;
                color: white;
                padding: 15px;
                border-radius: 5px;
                margin-top: 20px;
                display: none;
            }
            .model-badge {
                display: inline-block;
                padding: 2px 8px;
                border-radius: 3px;
                font-size: 11px;
                font-weight: 600;
                margin-left: 8px;
            }
            .badge-fast {
                background: #27ae60;
                color: white;
            }
            .badge-quality {
                background: #f39c12;
                color: white;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔬 BioMed-RAG</h1>
            <p class="subtitle">Biomedical Question Answering with Retrieval-Augmented Generation</p>
            
            <div class="input-group">
                <label for="question">Question</label>
                <textarea id="question" placeholder="What is the role of TP53 in cancer?"></textarea>
            </div>
            
            <div class="input-group">
                <label for="model">Model</label>
                <select id="model">
                    <option value="tinyllama">TinyLlama 1.1B <span class="model-badge badge-fast">⚡ FASTEST</span></option>
                    <option value="phi3">Phi-3 Mini (3.8B)</option>
                    <option value="llama">Llama 3.2 (3B)</option>
                    <option value="biomistral">BioMistral (7B) <span class="model-badge badge-quality">🏆 BEST QUALITY</span></option>
                </select>
            </div>
            
            <div class="input-group">
                <label>
                    <input type="checkbox" id="use_rag" checked> Use Retrieval-Augmented Generation
                </label>
            </div>
            
            <button onclick="askQuestion()" id="submitBtn">Ask Question</button>
            
            <div id="loading" class="loading">
                🔄 Generating answer... This may take 15-60 seconds on CPU.
            </div>
            
            <div id="error" class="error">
                <strong>Error:</strong>
                <div id="error-message"></div>
            </div>
            
            <div id="answer-box" class="answer-box">
                <h3>Answer:</h3>
                <p id="answer-text"></p>
                
                <div class="meta-info">
                    <div class="meta-item">
                        <span class="meta-label">Model</span>
                        <span id="meta-model"></span>
                    </div>
                    <div class="meta-item">
                        <span class="meta-label">Total Time</span>
                        <span id="meta-time"></span>
                    </div>
                    <div class="meta-item">
                        <span class="meta-label">Documents</span>
                        <span id="meta-docs"></span>
                    </div>
                    <div class="meta-item">
                        <span class="meta-label">PMIDs</span>
                        <span id="meta-pmids"></span>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            async function askQuestion() {
                const question = document.getElementById('question').value;
                const model = document.getElementById('model').value;
                const use_rag = document.getElementById('use_rag').checked;
                
                if (!question.trim()) {
                    alert('Please enter a question');
                    return;
                }
                
                // Show loading
                document.getElementById('submitBtn').disabled = true;
                document.getElementById('loading').style.display = 'block';
                document.getElementById('answer-box').classList.remove('show');
                document.getElementById('error').style.display = 'none';
                
                try {
                    const response = await fetch('/answer', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            question: question,
                            model: model,
                            use_rag: use_rag,
                            n_docs: 5,
                            temperature: 0.7,
                            max_tokens: 256
                        })
                    });
                    
                    const data = await response.json();
                    
                    if (!response.ok) {
                        throw new Error(data.detail || 'Failed to get answer');
                    }
                    
                    // Display answer
                    document.getElementById('answer-text').textContent = data.answer;
                    document.getElementById('meta-model').textContent = data.model.toUpperCase();
                    document.getElementById('meta-time').textContent = data.total_time.toFixed(2) + 's';
                    document.getElementById('meta-docs').textContent = data.num_docs_retrieved;
                    
                    const pmids = data.retrieved_pmids || [];
                    if (pmids.length > 0) {
                        document.getElementById('meta-pmids').textContent = 
                            pmids.slice(0, 3).join(', ') + (pmids.length > 3 ? '...' : '');
                    } else {
                        document.getElementById('meta-pmids').textContent = 'N/A';
                    }
                    
                    document.getElementById('answer-box').classList.add('show');
                    
                } catch (error) {
                    console.error('Error:', error);
                    document.getElementById('error-message').textContent = error.message;
                    document.getElementById('error').style.display = 'block';
                } finally {
                    document.getElementById('submitBtn').disabled = false;
                    document.getElementById('loading').style.display = 'none';
                }
            }
            
            // Allow Enter to submit (with Shift+Enter for newline)
            document.getElementById('question').addEventListener('keydown', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    askQuestion();
                }
            });
        </script>
    </body>
    </html>
    """
    return html_content


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    faiss_connected = False
    try:
        import requests
        response = requests.get("http://localhost:5000/health", timeout=2)
        faiss_connected = response.status_code == 200
    except Exception as e:
        logger.warning(f"FAISS health check failed: {e}")
    
    return {
        "status": "healthy",
        "available_models": list(MODEL_INFO.keys()),
        "faiss_connected": faiss_connected
    }


@app.get("/models")
async def list_models():
    """List available models and their specifications."""
    return MODEL_INFO


@app.post("/answer", response_model=QuestionResponse)
async def answer_question(request: QuestionRequest):
    """Answer a biomedical question."""
    logger.info(f"Received question: {request.question[:100]}...")
    logger.info(f"Model: {request.model}, Use RAG: {request.use_rag}")
    
    # Validate model
    if request.model not in MODEL_INFO:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model. Choose from: {list(MODEL_INFO.keys())}"
        )
    
    try:
        # Get RAG instance
        logger.info("Getting RAG instance...")
        rag = get_rag_instance(request.model, request.use_rag)
        rag.n_docs = request.n_docs
        
        # Generate answer
        logger.info("Generating answer...")
        result = rag.answer_question(
            question=request.question,
            temperature=request.temperature,
            max_new_tokens=request.max_tokens,
            return_context=False
        )
        
        logger.info(f"Answer generated successfully in {result['total_time']:.2f}s")
        
        return QuestionResponse(**result)
        
    except Exception as e:
        logger.error(f"Error in answer_question: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}\n\nCheck server logs for details."
        )


@app.post("/compare")
async def compare_models(question: str, models: Optional[List[str]] = None):
    """Compare all models on the same question."""
    if models is None:
        models = ["tinyllama", "phi3"]  # Default to fast models
    
    results = {}
    
    for model_name in models:
        try:
            logger.info(f"Comparing with model: {model_name}")
            rag = get_rag_instance(model_name, use_rag=True)
            result = rag.answer_question(question)
            results[model_name] = result
        except Exception as e:
            logger.error(f"Error with model {model_name}: {e}")
            results[model_name] = {"error": str(e)}
    
    return results


@app.post("/ablation")
async def ablation_study(question: str, model: str = "tinyllama"):
    """Compare RAG vs No-RAG for a single model."""
    try:
        logger.info(f"Running ablation study with {model}")
        rag = get_rag_instance(model, use_rag=True)
        comparison = rag.compare_rag_vs_no_rag(question)
        return comparison
    except Exception as e:
        logger.error(f"Error in ablation study: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 Starting BioMed-RAG API Server")
    print("=" * 60)
    print("\nAvailable models:")
    for name, info in MODEL_INFO.items():
        print(f"  - {name}: {info['name']} ({info['parameters']})")
    print("\n" + "=" * 60)
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🌐 Web Interface: http://localhost:8000")
    print("=" * 60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")