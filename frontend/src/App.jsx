import { useState } from "react";
import axios from "axios";
import "./App.css";

const API = "http://localhost:8000";
const THRESHOLD = 0.75;

// Cycled through by the "Load example" button in TryItOut.
const EXAMPLES = [
  { a: "Write unit tests for auth module", b: "Set up CI/CD pipeline" },
  { a: "Redesign the landing page", b: "Update the marketing site homepage" },
  { a: "Fix login button not responding", b: "Login button is broken" },
  { a: "Add password reset flow", b: "Implement forgot password functionality" },
];

// Supported task boards for the Connect tab.
const PLATFORMS = [
  { key: "trello", name: "Trello", logo: "https://cdn.worldvectorlogo.com/logos/trello.svg" },
  { key: "jira", name: "Jira", logo: "https://cdn.worldvectorlogo.com/logos/jira-1.svg" },
  { key: "asana", name: "Asana", logo: "https://cdn.worldvectorlogo.com/logos/asana-logo.svg" },
];

function ConfidenceBadge({ similarity }) {
  const pct = similarity * 100;
  const { level, label } =
    pct >= 85
      ? { level: "high", label: "High" }
      : pct >= 70
        ? { level: "medium", label: "Medium" }
        : { level: "low", label: "Low" };
  return <span className={`badge badge--${level}`}>{label}</span>;
}

function App() {
  const [tab, setTab] = useState("about");

  return (
    <div className="app">
      <header className="header">
        <h1>Prune</h1>
        <p className="subtitle">Stop doing the same work twice.</p>
      </header>

      <nav className="tabs">
        <button
          className={tab === "about" ? "tab tab--active" : "tab"}
          onClick={() => setTab("about")}
        >
          About
        </button>
        <button
          className={tab === "try" ? "tab tab--active" : "tab"}
          onClick={() => setTab("try")}
        >
          Try it out
        </button>
        <button
          className={tab === "trello" ? "tab tab--active" : "tab"}
          onClick={() => setTab("trello")}
        >
          Connect Your Task Board
        </button>
      </nav>

      {tab === "try" && <TryItOut />}
      {tab === "trello" && <ConnectBoard />}
      {tab === "about" && <About />}
    </div>
  );
}

function TryItOut() {
  const [taskA, setTaskA] = useState("");
  const [taskB, setTaskB] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [exampleIndex, setExampleIndex] = useState(-1);

  // Demo board state — moved here from the Connect tab.
  const [demoLoading, setDemoLoading] = useState(false);
  const [demoError, setDemoError] = useState("");
  const [demoRan, setDemoRan] = useState(false);
  const [demoPairs, setDemoPairs] = useState([]); // { id, taskA, taskB, similarity, pending }

  const check = async (a = taskA, b = taskB) => {
    setError("");
    setResult(null);
    setLoading(true);
    try {
      const { data } = await axios.post(`${API}/check`, {
        task_a: a,
        task_b: b,
      });
      setResult(data);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  const goExample = (index) => {
    const i = ((index % EXAMPLES.length) + EXAMPLES.length) % EXAMPLES.length;
    const ex = EXAMPLES[i];
    setTaskA(ex.a);
    setTaskB(ex.b);
    setExampleIndex(i);
    check(ex.a, ex.b);
  };

  const loadDemo = async () => {
    setDemoError("");
    setDemoPairs([]);
    setDemoRan(false);
    setDemoLoading(true);
    try {
      const { data } = await axios.get(`${API}/demo`);
      const tasks = data.tasks || [];

      // Every unique pair (i < j) through the classifier.
      const requests = [];
      for (let i = 0; i < tasks.length; i++) {
        for (let j = i + 1; j < tasks.length; j++) {
          requests.push([tasks[i], tasks[j]]);
        }
      }

      const results = await Promise.all(
        requests.map(([a, b]) =>
          axios
            .post(`${API}/check`, { task_a: a.name, task_b: b.name })
            .then(({ data }) => ({ a, b, ...data }))
        )
      );

      const flagged = results
        .filter((r) => r.is_duplicate)
        .map((r) => ({
          id: `${r.a.id}-${r.b.id}`,
          taskA: r.a.name,
          taskB: r.b.name,
          similarity: r.similarity,
          pending: false,
        }))
        .sort((x, y) => y.similarity - x.similarity);

      setDemoPairs(flagged);
      setDemoRan(true);
    } catch (err) {
      setDemoError(errorMessage(err));
    } finally {
      setDemoLoading(false);
    }
  };

  const sendDemoFeedback = async (pair, label) => {
    setDemoPairs((prev) =>
      prev.map((p) => (p.id === pair.id ? { ...p, pending: true } : p))
    );
    try {
      await axios.post(`${API}/feedback`, {
        task_a: pair.taskA,
        task_b: pair.taskB,
        label,
      });
      // Reviewed pairs leave the list.
      setDemoPairs((prev) => prev.filter((p) => p.id !== pair.id));
    } catch (err) {
      setDemoError(errorMessage(err));
      setDemoPairs((prev) =>
        prev.map((p) => (p.id === pair.id ? { ...p, pending: false } : p))
      );
    }
  };

  const disabled = loading || !taskA.trim() || !taskB.trim();

  return (
    <section className="panel">
      <label className="field">
        <span className="field__label">Task A</span>
        <input
          className="input"
          value={taskA}
          onChange={(e) => setTaskA(e.target.value)}
          placeholder="e.g. Fix the broken login button"
        />
      </label>

      <label className="field">
        <span className="field__label">Task B</span>
        <input
          className="input"
          value={taskB}
          onChange={(e) => setTaskB(e.target.value)}
          placeholder="e.g. Login button is not working"
        />
      </label>

      <div className="example">
        {exampleIndex < 0 ? (
          <button
            className="btn btn--ghost"
            onClick={() => goExample(0)}
            disabled={loading}
          >
            Load example
          </button>
        ) : (
          <>
            <div style={{ display: "flex", gap: "8px" }}>
              <button
                className="btn btn--ghost"
                onClick={() => goExample(exampleIndex - 1)}
                disabled={loading}
              >
                ← Back
              </button>
              <button
                className="btn btn--ghost"
                onClick={() => goExample(exampleIndex + 1)}
                disabled={loading}
              >
                Next →
              </button>
            </div>
            <span className="example__label">
              Example {exampleIndex + 1} of {EXAMPLES.length}
            </span>
          </>
        )}
      </div>

      <button
        className="btn btn--primary"
        onClick={() => check()}
        disabled={disabled}
      >
        {loading ? "Checking…" : "Check"}
      </button>

      {error && <div className="error">{error}</div>}

      {result && (
        <div className="result-card">
          <div className="result-card__row">
            <span className="result-card__label">Similarity</span>
            <span className="score">{result.similarity.toFixed(4)}</span>
          </div>
          <div className="result-card__row">
            <span className="result-card__label">Verdict</span>
            <span
              className={
                result.is_duplicate
                  ? "verdict verdict--dup"
                  : "verdict verdict--unique"
              }
            >
              {result.is_duplicate ? "Duplicate" : "Not a duplicate"}
            </span>
          </div>
          {result.overridden && (
            <div
              style={{
                color: "var(--muted)",
                fontSize: "12px",
                fontStyle: "italic",
              }}
            >
              {"Similarity too low to be a duplicate (< 60%)"}
            </div>
          )}
          <div className="result-card__confidence">
            <span className="result-card__label">Confidence</span>
            <div className="confbar">
              <div className="confbar__track">
                <div
                  className={
                    result.is_duplicate
                      ? "confbar__fill confbar__fill--dup"
                      : "confbar__fill confbar__fill--unique"
                  }
                  style={{ width: `${(result.confidence * 100).toFixed(1)}%` }}
                />
              </div>
              <span className="confbar__pct">
                {(result.confidence * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        </div>
      )}

      <div style={{ borderTop: "1px solid var(--border)" }} />

      <div className="demo" style={{ alignItems: "center", textAlign: "center" }}>
        <span className="demo__hint">Want to try it on a full task board?</span>
        <button
          className="btn btn--ghost"
          style={{ alignSelf: "center" }}
          onClick={loadDemo}
          disabled={demoLoading}
        >
          {demoLoading ? "Analyzing board…" : "Load Demo Board"}
        </button>
      </div>

      {demoError && <div className="error">{demoError}</div>}

      {demoPairs.length > 0 && (
        <div className="pairs">
          {demoPairs.map((pair) => (
            <div key={pair.id} className="pair">
              <div className="pair__tasks">
                <div className="pair__task">{pair.taskA}</div>
                <div className="pair__vs">vs</div>
                <div className="pair__task">{pair.taskB}</div>
              </div>

              <div className="pair__footer">
                <div className="pair__score">
                  <span className="score">
                    {(pair.similarity * 100).toFixed(1)}%
                  </span>
                  <ConfidenceBadge similarity={pair.similarity} />
                </div>

                <div className="pair__actions">
                  <button
                    className="btn btn--confirm"
                    disabled={pair.pending}
                    onClick={() => sendDemoFeedback(pair, 1)}
                  >
                    {pair.pending ? "…" : "Confirm"}
                  </button>
                  <button
                    className="btn btn--dismiss"
                    disabled={pair.pending}
                    onClick={() => sendDemoFeedback(pair, 0)}
                  >
                    {pair.pending ? "…" : "Dismiss"}
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {demoRan && !demoLoading && demoPairs.length === 0 && !demoError && (
        <div className="demo__empty">No flagged pairs left to review.</div>
      )}
    </section>
  );
}

function ConnectBoard() {
  const [platform, setPlatform] = useState(null); // "trello" | "jira" | "asana"

  // Trello credentials.
  const [apiKey, setApiKey] = useState("");
  const [token, setToken] = useState("");
  const [boardId, setBoardId] = useState("");

  // Jira credentials.
  const [domain, setDomain] = useState("");
  const [email, setEmail] = useState("");
  const [jiraToken, setJiraToken] = useState("");
  const [projectKey, setProjectKey] = useState("");

  // Asana credentials.
  const [asanaToken, setAsanaToken] = useState("");
  const [projectId, setProjectId] = useState("");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [tasksScanned, setTasksScanned] = useState(0);
  const [boardName, setBoardName] = useState("");
  const [pairs, setPairs] = useState([]); // { id, taskA, taskB, similarity, status, pending }

  const connect = async () => {
    setError("");
    setPairs([]);
    setTasksScanned(0);
    setLoading(true);
    try {
      // Pick the endpoint + body for the selected platform.
      const config = {
        trello: {
          endpoint: "/trello",
          body: { api_key: apiKey, token, board_id: boardId },
        },
        jira: {
          endpoint: "/jira",
          body: {
            domain,
            email,
            api_token: jiraToken,
            project_key: projectKey,
          },
        },
        asana: {
          endpoint: "/asana",
          body: { access_token: asanaToken, project_id: projectId },
        },
      }[platform];

      const { data } = await axios.post(`${API}${config.endpoint}`, config.body);
      const tasks = data.tasks || [];

      const name = PLATFORMS.find((p) => p.key === platform)?.name || "";
      setBoardName(data.board_name ? `${name} · ${data.board_name}` : name);

      // One vectorized scan instead of O(n^2) /check round-trips.
      const { data: scan } = await axios.post(`${API}/scan`, {
        tasks,
        threshold: THRESHOLD,
      });

      setTasksScanned(scan.tasks_scanned);
      setPairs(
        (scan.pairs || []).map((p) => ({
          id: p.id,
          taskA: p.task_a,
          taskB: p.task_b,
          similarity: p.similarity,
          status: null, // null | "confirmed" | "dismissed"
          pending: false,
        }))
      );
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  const sendFeedback = async (pair, label) => {
    setPairs((prev) =>
      prev.map((p) => (p.id === pair.id ? { ...p, pending: true } : p))
    );
    try {
      await axios.post(`${API}/feedback`, {
        task_a: pair.taskA,
        task_b: pair.taskB,
        label,
      });
      setPairs((prev) =>
        prev.map((p) =>
          p.id === pair.id
            ? {
                ...p,
                pending: false,
                status: label === 1 ? "confirmed" : "dismissed",
              }
            : p
        )
      );
    } catch (err) {
      setError(errorMessage(err));
      setPairs((prev) =>
        prev.map((p) => (p.id === pair.id ? { ...p, pending: false } : p))
      );
    }
  };

  const newBoard = () => {
    setPlatform(null);
    setApiKey("");
    setToken("");
    setBoardId("");
    setDomain("");
    setEmail("");
    setJiraToken("");
    setProjectKey("");
    setAsanaToken("");
    setProjectId("");
    setPairs([]);
    setTasksScanned(0);
    setBoardName("");
    setError("");
  };

  const confirmedCount = pairs.filter((p) => p.status === "confirmed").length;
  const connected = tasksScanned > 0;
  const credsFilled =
    platform === "trello"
      ? apiKey.trim() && token.trim() && boardId.trim()
      : platform === "jira"
        ? domain.trim() && email.trim() && jiraToken.trim() && projectKey.trim()
        : platform === "asana"
          ? asanaToken.trim() && projectId.trim()
          : false;
  const disabled = loading || !credsFilled;

  return (
    <section className="panel">
      {connected ? (
        <button className="link" onClick={newBoard}>
          ← New board
        </button>
      ) : (
        <>
          <div className="platforms">
            {PLATFORMS.map((p) => (
              <button
                key={p.key}
                type="button"
                className={
                  platform === p.key
                    ? "platform platform--selected"
                    : "platform"
                }
                onClick={() => setPlatform(p.key)}
              >
                <img
                  className="platform__logo"
                  src={p.logo}
                  alt={`${p.name} logo`}
                />
                <span className="platform__name">{p.name}</span>
              </button>
            ))}
          </div>

          {platform === "trello" && (
            <>
              <label className="field">
                <span className="field__label">API Key</span>
                <input
                  className="input"
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  placeholder="Your Trello API key"
                />
              </label>
              <label className="field">
                <span className="field__label">Token</span>
                <input
                  className="input"
                  value={token}
                  onChange={(e) => setToken(e.target.value)}
                  placeholder="Your Trello token"
                />
              </label>
              <label className="field">
                <span className="field__label">Board ID</span>
                <input
                  className="input"
                  value={boardId}
                  onChange={(e) => setBoardId(e.target.value)}
                  placeholder="Board ID"
                />
              </label>
            </>
          )}

          {platform === "jira" && (
            <>
              <label className="field">
                <span className="field__label">Domain</span>
                <input
                  className="input"
                  value={domain}
                  onChange={(e) => setDomain(e.target.value)}
                  placeholder="yourcompany.atlassian.net"
                />
              </label>
              <label className="field">
                <span className="field__label">Email</span>
                <input
                  className="input"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="you@company.com"
                />
              </label>
              <label className="field">
                <span className="field__label">API Token</span>
                <input
                  className="input"
                  value={jiraToken}
                  onChange={(e) => setJiraToken(e.target.value)}
                  placeholder="Your Jira API token"
                />
              </label>
              <label className="field">
                <span className="field__label">Project Key</span>
                <input
                  className="input"
                  value={projectKey}
                  onChange={(e) => setProjectKey(e.target.value)}
                  placeholder="e.g. MY-PROJECT"
                />
              </label>
            </>
          )}

          {platform === "asana" && (
            <>
              <label className="field">
                <span className="field__label">Personal Access Token</span>
                <input
                  className="input"
                  value={asanaToken}
                  onChange={(e) => setAsanaToken(e.target.value)}
                  placeholder="Your Asana access token"
                />
              </label>
              <label className="field">
                <span className="field__label">Project ID</span>
                <input
                  className="input"
                  value={projectId}
                  onChange={(e) => setProjectId(e.target.value)}
                  placeholder="Project ID"
                />
              </label>
            </>
          )}

          {platform && (
            <button
              className="btn btn--primary"
              onClick={connect}
              disabled={disabled}
            >
              {loading ? "Scanning…" : "Connect"}
            </button>
          )}
        </>
      )}

      {error && <div className="error">{error}</div>}

      {connected && (
        <div
          style={{
            alignSelf: "flex-start",
            color: "var(--muted)",
            fontSize: "13px",
            border: "1px solid var(--border)",
            borderRadius: "999px",
            padding: "4px 12px",
          }}
        >
          {boardName || "Board connected"}
        </div>
      )}

      {(tasksScanned > 0 || pairs.length > 0) && (
        <div className="metrics">
          <div className="metric">
            <span className="metric__value">{tasksScanned}</span>
            <span className="metric__label">Tasks scanned</span>
          </div>
          <div className="metric">
            <span className="metric__value">{pairs.length}</span>
            <span className="metric__label">Pairs flagged</span>
          </div>
          <div className="metric">
            <span className="metric__value">{confirmedCount}</span>
            <span className="metric__label">Confirmed duplicates</span>
          </div>
        </div>
      )}

      <div className="pairs">
        {pairs.map((pair) => (
          <div
            key={pair.id}
            className={pair.status ? "pair pair--resolved" : "pair"}
          >
            <div className="pair__tasks">
              <div className="pair__task">{pair.taskA}</div>
              <div className="pair__vs">vs</div>
              <div className="pair__task">{pair.taskB}</div>
            </div>

            <div className="pair__footer">
              <div className="pair__score">
                <span className="score">
                  {(pair.similarity * 100).toFixed(1)}%
                </span>
                <ConfidenceBadge similarity={pair.similarity} />
              </div>

              {pair.status ? (
                <span
                  className={
                    pair.status === "confirmed"
                      ? "status status--confirmed"
                      : "status status--dismissed"
                  }
                >
                  {pair.status === "confirmed" ? "Confirmed" : "Dismissed"}
                </span>
              ) : (
                <div className="pair__actions">
                  <button
                    className="btn btn--confirm"
                    disabled={pair.pending}
                    onClick={() => sendFeedback(pair, 1)}
                  >
                    {pair.pending ? "…" : "Confirm"}
                  </button>
                  <button
                    className="btn btn--dismiss"
                    disabled={pair.pending}
                    onClick={() => sendFeedback(pair, 0)}
                  >
                    {pair.pending ? "…" : "Dismiss"}
                  </button>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function About() {
  return (
    <section className="about">
      {/* Pull quote */}
      <div className="about__quote">
        <p className="about__quote-text">
          “The average employee spends 4 hours and 38 minutes on duplicate
          tasks daily.”
        </p>
        <p className="about__quote-cite">— Asana</p>
      </div>

      {/* Stat cards */}
      <div className="about__stats">
        <div className="stat-card">
          <span className="stat-card__num">4h 38m</span>
          <span className="stat-card__label">wasted daily on duplicate work</span>
        </div>
        <div className="stat-card">
          <span className="stat-card__num">33%</span>
          <span className="stat-card__label">
            of teams lack cross-team coordination
          </span>
        </div>
        <div className="stat-card">
          <span className="stat-card__num">55B</span>
          <span className="stat-card__label">hours lost globally each year</span>
        </div>
      </div>

      {/* How it works */}
      <div className="about__how">
        <p className="about__how-lead">
          Prune catches duplicates before they cost you.
        </p>
        <div className="steps">
          <div className="step">
            <span className="step__title">① Pull</span>
            <span className="step__desc">
              Connect your task board and fetch all tasks
            </span>
          </div>
          <span className="step__arrow">→</span>
          <div className="step">
            <span className="step__title">② Compare</span>
            <span className="step__desc">
              ML embeddings find semantic matches
            </span>
          </div>
          <span className="step__arrow">→</span>
          <div className="step">
            <span className="step__title">③ Learn</span>
            <span className="step__desc">Your feedback improves the model</span>
          </div>
        </div>
      </div>

      {/* Closing */}
      <p className="about__closing">
        Built with an active learning loop.
      </p>
    </section>
  );
}

function errorMessage(err) {
  if (err.response) {
    const detail = err.response.data?.detail;
    return detail
      ? `Error: ${detail}`
      : `Request failed (${err.response.status}).`;
  }
  return "Could not reach the backend. Is it running on http://localhost:8000?";
}

export default App;
