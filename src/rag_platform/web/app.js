const messages = document.querySelector("#messages");
const queryForm = document.querySelector("#query-form");
const uploadForm = document.querySelector("#upload-form");
const questionInput = document.querySelector("#question");
const uploadStatus = document.querySelector("#upload-status");
const chunkCount = document.querySelector("#chunk-count");
const latency = document.querySelector("#latency");

function addMessage(role, text, citations = []) {
  const article = document.createElement("article");
  article.className = `message ${role}`;
  const paragraph = document.createElement("p");
  paragraph.textContent = text;
  article.appendChild(paragraph);

  if (citations.length) {
    const citationList = document.createElement("div");
    citationList.className = "citations";
    citations.slice(0, 5).forEach((citation, index) => {
      const item = document.createElement("span");
      const source = citation.metadata?.filename || citation.metadata?.source || citation.document_id;
      item.textContent = `[${index + 1}] ${source} | score ${citation.score}`;
      citationList.appendChild(item);
    });
    article.appendChild(citationList);
  }

  messages.appendChild(article);
  messages.scrollTop = messages.scrollHeight;
}

async function refreshHealth() {
  const response = await fetch("/healthz");
  const payload = await response.json();
  chunkCount.textContent = payload.chunks;
}

queryForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  addMessage("user", question);
  questionInput.value = "";

  const response = await fetch("/query", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ question, top_k: 5 }),
  });
  const payload = await response.json();
  if (!response.ok) {
    addMessage("assistant", payload.detail || "The assistant could not answer that request.");
    return;
  }

  latency.textContent = `${payload.latency_ms}ms`;
  addMessage("assistant", payload.answer, payload.citations);
});

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const input = document.querySelector("#files");
  if (!input.files.length) return;

  const body = new FormData();
  Array.from(input.files).forEach((file) => body.append("files", file));
  uploadStatus.textContent = "Indexing files...";

  const response = await fetch("/ingest/files", { method: "POST", body });
  const payload = await response.json();
  if (!response.ok) {
    uploadStatus.textContent = payload.detail || "Upload failed.";
    return;
  }
  uploadStatus.textContent = `Indexed ${payload.indexed_documents} documents into ${payload.indexed_chunks} chunks.`;
  await refreshHealth();
});

refreshHealth();
