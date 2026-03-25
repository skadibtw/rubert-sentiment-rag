DEMO_PAGE_HTML = """
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>NLP Pet Project Demo</title>
    <style>
      :root {
        --bg: #f4efe6;
        --panel: rgba(255, 250, 242, 0.92);
        --card: rgba(255, 255, 255, 0.78);
        --ink: #1f2933;
        --muted: #52606d;
        --accent: #ad4f2d;
        --accent-soft: #f6d6c5;
        --accent-alt: #22543d;
        --accent-alt-soft: rgba(34, 84, 61, 0.12);
        --border: rgba(31, 41, 51, 0.12);
        --shadow: 0 20px 60px rgba(75, 85, 99, 0.15);
      }

      * {
        box-sizing: border-box;
      }

      body {
        margin: 0;
        min-height: 100vh;
        font-family: Georgia, "Times New Roman", serif;
        color: var(--ink);
        background:
          radial-gradient(circle at top left, rgba(173, 79, 45, 0.18), transparent 34%),
          radial-gradient(circle at bottom right, rgba(53, 122, 93, 0.16), transparent 30%),
          linear-gradient(135deg, #efe7da 0%, #f8f4ec 45%, #efe3d6 100%);
        padding: 24px;
      }

      .shell {
        width: min(1180px, 100%);
        margin: 0 auto;
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 28px;
        box-shadow: var(--shadow);
        overflow: hidden;
        backdrop-filter: blur(16px);
      }

      .hero {
        padding: 32px 32px 18px;
        border-bottom: 1px solid var(--border);
      }

      .eyebrow {
        text-transform: uppercase;
        letter-spacing: 0.16em;
        font-size: 12px;
        color: var(--accent);
        margin: 0 0 12px;
      }

      h1 {
        margin: 0;
        font-size: clamp(32px, 5vw, 56px);
        line-height: 0.96;
        max-width: 12ch;
      }

      .subtitle {
        margin: 16px 0 0;
        max-width: 70ch;
        font-size: 18px;
        line-height: 1.6;
        color: var(--muted);
      }

      .status-row {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin-top: 20px;
      }

      .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 10px 12px;
        border-radius: 999px;
        font-size: 14px;
        background: var(--accent-alt-soft);
        color: var(--accent-alt);
      }

      .status-pill.error {
        background: rgba(173, 79, 45, 0.14);
        color: #8c3c1d;
      }

      .content {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 20px;
        padding: 24px 32px 32px;
      }

      .card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 22px;
        padding: 22px;
      }

      .card h2 {
        margin: 0 0 8px;
        font-size: 28px;
      }

      .card p {
        margin-top: 0;
      }

      label {
        display: block;
        font-size: 14px;
        margin: 14px 0 10px;
        color: var(--muted);
      }

      textarea,
      select,
      input {
        width: 100%;
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 14px 16px;
        font: inherit;
        font-size: 15px;
        background: rgba(255, 255, 255, 0.94);
        color: var(--ink);
      }

      textarea {
        min-height: 180px;
        resize: vertical;
      }

      .controls {
        display: grid;
        grid-template-columns: 1fr 120px 160px;
        gap: 12px;
        margin-top: 12px;
      }

      .actions {
        display: flex;
        gap: 12px;
        margin-top: 16px;
      }

      button {
        border: 0;
        border-radius: 999px;
        padding: 14px 20px;
        font: inherit;
        font-size: 15px;
        cursor: pointer;
        color: #fff;
        background: linear-gradient(135deg, #ad4f2d 0%, #8c3c1d 100%);
        box-shadow: 0 12px 24px rgba(173, 79, 45, 0.24);
      }

      button.secondary {
        background: linear-gradient(135deg, #2f855a 0%, #276749 100%);
        box-shadow: 0 12px 24px rgba(39, 103, 73, 0.2);
      }

      .hint,
      .metric,
      .meta-line {
        font-size: 14px;
        color: var(--muted);
      }

      .result-label {
        margin: 0;
        font-size: 34px;
      }

      .scores,
      .contexts {
        display: grid;
        gap: 12px;
        margin-top: 18px;
      }

      .score-row {
        display: grid;
        grid-template-columns: 92px 1fr auto;
        gap: 12px;
        align-items: center;
      }

      .bar {
        height: 10px;
        border-radius: 999px;
        background: #eee4d8;
        overflow: hidden;
      }

      .bar > span {
        display: block;
        height: 100%;
        background: linear-gradient(90deg, var(--accent-soft), var(--accent));
      }

      .section-title {
        margin: 20px 0 8px;
        font-size: 16px;
      }

      .context-item {
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 14px;
        background: rgba(255, 255, 255, 0.7);
      }

      .context-item p {
        margin: 8px 0 0;
        color: var(--ink);
        line-height: 1.5;
      }

      .keyphrases {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 12px;
      }

      .chip {
        display: inline-flex;
        align-items: center;
        padding: 8px 12px;
        border-radius: 999px;
        background: rgba(173, 79, 45, 0.1);
        color: var(--accent);
        font-size: 13px;
      }

      .answer-box {
        margin-top: 12px;
        padding: 16px;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.76);
        border: 1px solid var(--border);
        line-height: 1.6;
      }

      @media (max-width: 980px) {
        .content {
          grid-template-columns: 1fr;
        }
      }

      @media (max-width: 720px) {
        body {
          padding: 12px;
        }

        .hero,
        .content {
          padding-left: 18px;
          padding-right: 18px;
        }

        .controls {
          grid-template-columns: 1fr;
        }

        .actions {
          flex-direction: column;
        }
      }
    </style>
  </head>
  <body>
    <main class="shell">
      <section class="hero">
        <p class="eyebrow">Russian Review Intelligence</p>
        <h1>Predict sentiment and ask the review corpus.</h1>
        <p class="subtitle">
          The demo combines ruBERT sentiment classification with a retrieval pipeline
          over RuReviews. Use the left panel for single-review inference and the
          right panel to ask corpus-level questions.
        </p>
        <div class="status-row">
          <div id="model-health" class="status-pill">Checking model...</div>
          <div id="rag-health" class="status-pill">Checking RAG index...</div>
        </div>
      </section>

      <section class="content">
        <article class="card">
          <h2>Sentiment</h2>
          <p class="hint">Manual single-text inference through <code>/predict</code>.</p>
          <label for="review">Review text</label>
          <textarea id="review">Приложение удобное, но после последнего обновления стало заметно медленнее.</textarea>
          <div class="actions">
            <button id="predict-button" type="button">Analyze sentiment</button>
          </div>

          <div class="section-title">Prediction</div>
          <p id="label" class="result-label">-</p>
          <p class="metric">Original text</p>
          <p id="echo" class="hint">-</p>
          <div id="scores" class="scores"></div>
        </article>

        <article class="card">
          <h2>Review QA</h2>
          <p class="hint">Retrieval QA through <code>/ask</code> with optional LLM synthesis.</p>
          <label for="question">Question about the review corpus</label>
          <textarea id="question">На что чаще всего жалуются после обновления?</textarea>

          <div class="controls">
            <div>
              <label for="sentiment-focus">Sentiment focus</label>
              <select id="sentiment-focus">
                <option value="">Auto</option>
                <option value="negative" selected>Negative</option>
                <option value="neutral">Neutral</option>
                <option value="positive">Positive</option>
              </select>
            </div>
            <div>
              <label for="top-k">Top K</label>
              <input id="top-k" type="number" min="1" max="20" value="5" />
            </div>
            <div>
              <label for="generation-mode">Generator</label>
              <select id="generation-mode">
                <option value="auto" selected>Auto</option>
                <option value="llm">LLM</option>
                <option value="extractive">Extractive</option>
              </select>
            </div>
          </div>

          <div class="actions">
            <button id="ask-button" class="secondary" type="button">Ask reviews</button>
          </div>

          <div class="section-title">Answer</div>
          <div id="answer" class="answer-box">-</div>
          <p id="answer-meta" class="meta-line">-</p>

          <div class="section-title">Key phrases</div>
          <div id="keyphrases" class="keyphrases"></div>

          <div class="section-title">Retrieved contexts</div>
          <div id="contexts" class="contexts"></div>
        </article>
      </section>
    </main>

    <script>
      const modelHealthNode = document.getElementById("model-health");
      const ragHealthNode = document.getElementById("rag-health");
      const labelNode = document.getElementById("label");
      const echoNode = document.getElementById("echo");
      const scoresNode = document.getElementById("scores");
      const reviewNode = document.getElementById("review");
      const predictButtonNode = document.getElementById("predict-button");
      const questionNode = document.getElementById("question");
      const sentimentFocusNode = document.getElementById("sentiment-focus");
      const topKNode = document.getElementById("top-k");
      const generationModeNode = document.getElementById("generation-mode");
      const askButtonNode = document.getElementById("ask-button");
      const answerNode = document.getElementById("answer");
      const answerMetaNode = document.getElementById("answer-meta");
      const keyphrasesNode = document.getElementById("keyphrases");
      const contextsNode = document.getElementById("contexts");

      function setHealth(node, readyText, missingText, isReady) {
        node.textContent = isReady ? readyText : missingText;
        node.className = isReady ? "status-pill" : "status-pill error";
      }

      function renderScores(scores) {
        const entries = Object.entries(scores).sort((a, b) => b[1] - a[1]);
        scoresNode.innerHTML = entries
          .map(([label, value]) => {
            const percent = (value * 100).toFixed(2);
            return `
              <div class="score-row">
                <strong>${label}</strong>
                <div class="bar"><span style="width: ${percent}%"></span></div>
                <span>${percent}%</span>
              </div>
            `;
          })
          .join("");
      }

      function renderKeyphrases(keyphrases) {
        keyphrasesNode.innerHTML = keyphrases.length
          ? keyphrases.map((item) => `<span class="chip">${item}</span>`).join("")
          : `<span class="hint">No key phrases extracted.</span>`;
      }

      function renderContexts(contexts) {
        contextsNode.innerHTML = contexts.length
          ? contexts
              .map((item) => {
                return `
                  <div class="context-item">
                    <div class="meta-line">${item.label} · ${item.split} · score ${item.score.toFixed(4)}</div>
                    <p>${item.text}</p>
                  </div>
                `;
              })
              .join("")
          : `<div class="hint">No contexts returned.</div>`;
      }

      async function checkHealth() {
        try {
          const response = await fetch("/health");
          const payload = await response.json();
          setHealth(
            modelHealthNode,
            `Model ready: ${payload.model_dir}`,
            `Model missing: ${payload.model_dir}`,
            payload.model_available,
          );
          setHealth(
            ragHealthNode,
            `RAG ready: ${payload.rag_index_dir}`,
            `RAG missing: ${payload.rag_index_dir}`,
            payload.rag_index_available,
          );
        } catch (error) {
          modelHealthNode.textContent = "Health check failed";
          modelHealthNode.className = "status-pill error";
          ragHealthNode.textContent = "Health check failed";
          ragHealthNode.className = "status-pill error";
        }
      }

      async function predict() {
        const text = reviewNode.value.trim();
        if (!text) {
          return;
        }

        predictButtonNode.disabled = true;
        predictButtonNode.textContent = "Analyzing...";

        try {
          const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text }),
          });
          const payload = await response.json();
          if (!response.ok) {
            throw new Error(payload.detail || "Prediction failed");
          }
          labelNode.textContent = payload.label;
          echoNode.textContent = payload.text;
          renderScores(payload.scores);
        } catch (error) {
          labelNode.textContent = "error";
          echoNode.textContent = String(error.message || error);
          scoresNode.innerHTML = "";
        } finally {
          predictButtonNode.disabled = false;
          predictButtonNode.textContent = "Analyze sentiment";
        }
      }

      async function askReviews() {
        const question = questionNode.value.trim();
        if (!question) {
          return;
        }

        askButtonNode.disabled = true;
        askButtonNode.textContent = "Searching...";

        try {
          const response = await fetch("/ask", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              question,
              top_k: Number(topKNode.value || 5),
              sentiment_focus: sentimentFocusNode.value || null,
              generation_mode: generationModeNode.value,
            }),
          });
          const payload = await response.json();
          if (!response.ok) {
            throw new Error(payload.detail || "RAG request failed");
          }
          answerNode.textContent = payload.answer;
          answerMetaNode.textContent = `Mode: ${payload.generation_mode}; LLM used: ${payload.llm_used}; Focus: ${payload.sentiment_focus || "auto"}`;
          renderKeyphrases(payload.keyphrases);
          renderContexts(payload.contexts);
        } catch (error) {
          answerNode.textContent = String(error.message || error);
          answerMetaNode.textContent = "-";
          keyphrasesNode.innerHTML = "";
          contextsNode.innerHTML = "";
        } finally {
          askButtonNode.disabled = false;
          askButtonNode.textContent = "Ask reviews";
        }
      }

      predictButtonNode.addEventListener("click", predict);
      askButtonNode.addEventListener("click", askReviews);
      checkHealth();
      predict();
      askReviews();
    </script>
  </body>
</html>
"""
