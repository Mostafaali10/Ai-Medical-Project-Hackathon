/**
 * Clinical RAG API Service Layer & Response Normalizer
 * Connects React frontend to FastAPI backend via Vite proxy or direct VITE_API_URL.
 */

// If VITE_API_URL is set, use it; otherwise use empty string so Vite dev server proxy handles it.
const RAW_URL = import.meta.env.VITE_API_URL;
const API_BASE_URL = RAW_URL ? RAW_URL.replace(/\/$/, '') : '';

async function handleResponse(response) {
  if (!response.ok) {
    let errorDetail = `HTTP ${response.status} ${response.statusText}`;
    try {
      const errorJson = await response.json();
      if (errorJson.message) {
        errorDetail = errorJson.message;
      } else if (errorJson.detail) {
        if (Array.isArray(errorJson.detail)) {
          errorDetail = errorJson.detail.map(d => d.msg || JSON.stringify(d)).join('; ');
        } else {
          errorDetail = typeof errorJson.detail === 'string' ? errorJson.detail : JSON.stringify(errorJson.detail);
        }
      } else if (errorJson.error) {
        errorDetail = errorJson.error;
      }
    } catch {
      // Non-JSON response fallback
    }

    if (response.status === 422) {
      errorDetail = `Validation Error (422): ${errorDetail}`;
    } else if (response.status === 503) {
      errorDetail = `Service Unavailable (503): ${errorDetail}`;
    } else if (response.status >= 500) {
      errorDetail = `Internal Server Error (${response.status}): ${errorDetail}`;
    }

    const err = new Error(errorDetail);
    err.status = response.status;
    throw err;
  }
  return response.json();
}

/**
 * Health check endpoint
 * @returns {Promise<{status: string, pipeline: string, vectorstore: string, llm: string, collection_name: string, indexed_chunks: number}>}
 */
export async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/health`);
    return await handleResponse(res);
  } catch (err) {
    console.error('[Health Check Failed]', err);
    if (err.name === 'TypeError' && err.message.toLowerCase().includes('fetch')) {
      throw new Error('Backend server is unreachable. Please verify FastAPI is running at port 8000.');
    }
    throw err;
  }
}

/**
 * Submit clinical question to RAG pipeline
 * @param {string} question 
 * @param {number} k 
 * @returns {Promise<object>} Raw AskResponse from backend
 */
export async function askQuestion(question, k = 5) {
  try {
    const res = await fetch(`${API_BASE_URL}/api/ask`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ question, k }),
    });
    return await handleResponse(res);
  } catch (err) {
    console.error('[Ask Question Failed]', err);
    if (err.name === 'TypeError' && err.message.toLowerCase().includes('fetch')) {
      throw new Error('Network or CORS error connecting to backend at ' + (API_BASE_URL || window.location.origin) + '. Please check server logs.');
    }
    throw err;
  }
}

/**
 * Fetch list of indexed documents
 * @returns {Promise<{documents: Array, collection_name: string, total_chunks: number}>}
 */
export async function fetchDocuments() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/documents`);
    return await handleResponse(res);
  } catch (err) {
    console.error('[Fetch Documents Failed]', err);
    if (err.name === 'TypeError' && err.message.toLowerCase().includes('fetch')) {
      throw new Error('Backend server is unreachable. Please verify FastAPI is running at port 8000.');
    }
    throw err;
  }
}

/**
 * Normalizes backend AskResponse into the UI model for ClinicalDashboard.jsx
 * @param {object} raw - Backend AskResponse object
 * @returns {object} Normalized view model
 */
export function normalizeAskResponse(raw) {
  if (!raw) return null;
  const answer = raw.answer || {};
  const citationReport = raw.citation_report || {};
  const chunksUsed = raw.chunks_used || [];

  return {
    raw, // Raw response for Raw JSON inspector tab
    status: answer.status || 'unknown',
    confidence: answer.confidence || 'Insufficient Evidence',
    schema_valid: Boolean(raw.schema_valid),
    recommendation: answer.recommendation || '',
    safety_note: answer.safety_note || '',
    missing_information: Array.isArray(answer.missing_information) ? answer.missing_information : [],
    supporting_evidence: (answer.supporting_evidence || []).map((item) => {
      const citations = Array.isArray(item.citations) ? item.citations : [];
      return {
        text: item.claim || '',
        cited: citations.join(', '),
        citations,
      };
    }),
    citation_validation: {
      valid: Boolean(citationReport.valid),
      invented: Array.isArray(citationReport.invented_citations) ? citationReport.invented_citations : [],
      claims_missing_citation: Array.isArray(citationReport.claims_missing_citation)
        ? citationReport.claims_missing_citation
        : [],
    },
    evidence_chunks: chunksUsed.map((c) => ({
      rank: c.rank || 0,
      score: typeof c.similarity_score === 'number' ? c.similarity_score : 0,
      doc_id: c.document_name || c.document_id || 'Unknown Document',
      page_number: c.page_number ?? 'N/A',
      section: c.section || 'General',
      chunk_id: c.chunk_id || 'N/A',
      content: c.text || '',
    })),
  };
}
