import { useEffect, useMemo, useRef, useState } from "react";

const API_BASE = "http://127.0.0.1:8000";
const API_URL = `${API_BASE}/predict`;

const CLASS_ORDER = [
  "Potato___Early_blight",
  "Potato___Late_blight",
  "Potato___healthy",
];

export default function App() {
  /* --------- THEME (Dark / Light) --------- */
  const getSystemPref = () =>
    window.matchMedia &&
    window.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light";

  const [theme, setTheme] = useState(
    () => localStorage.getItem("theme") || getSystemPref()
  );

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("theme", theme);
  }, [theme]);

  const toggleTheme = () => setTheme((t) => (t === "dark" ? "light" : "dark"));

  /* --------- State για εικόνα & αποτέλεσμα --------- */
  const [file, setFile] = useState(null);
  const [previewURL, setPreviewURL] = useState("");
  const [drag, setDrag] = useState(false);

  const [result, setResult] = useState(null); // { label, confidence, probs, shap_image }
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [shapImage, setShapImage] = useState(null);

  const inputRef = useRef(null);

  // όταν αλλάζει το file, φτιάχνουμε preview URL
  useEffect(() => {
    if (!file) {
      setPreviewURL("");
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewURL(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  /* --------- Handlers για drop / click --------- */
  const onPick = () => inputRef.current?.click();

  const onFileChange = (e) => {
    const f = e.target.files?.[0];
    if (f) setFile(f);
  };

  const onDragOver = (e) => {
    e.preventDefault();
    setDrag(true);
  };
  const onDragLeave = () => setDrag(false);
  const onDrop = (e) => {
    e.preventDefault();
    setDrag(false);
    const f = e.dataTransfer.files?.[0];
    if (f) setFile(f);
  };

  const onReset = () => {
    setFile(null);
    setResult(null);
    setPreviewURL("");
    setError(null);
    setShapImage(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  /* --------- Κλήση API --------- */
  const onPredict = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);
    setResult(null);
    setShapImage(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const text = await response.text();
        console.error("Backend error:", response.status, text);
        throw new Error("Request failed");
      }

      const data = await response.json();
      console.log("Prediction:", data);
      setResult(data);
      setShapImage(data.gradcam_image || null);
    } catch (err) {
      console.error("Predict error:", err);
      setError("Κάτι πήγε στραβά κατά την πρόβλεψη.");
    } finally {
      setLoading(false);
    }
  };

  const prettyPercent = (x) => `${(x * 100).toFixed(2)}%`;

  const bars = useMemo(() => {
    if (!result?.probs) return [];
    const entries = Object.entries(result.probs);
    // ταξινόμηση με βάση το CLASS_ORDER
    const byOrder = CLASS_ORDER
      .map((k) => entries.find(([label]) => label === k))
      .filter(Boolean);
    return byOrder;
  }, [result]);

  const barClass = (label) => {
    if (label.includes("Early")) return "bar-early";
    if (label.includes("Late")) return "bar-late";
    if (label.includes("healthy")) return "bar-healthy";
    return "";
  };

  return (
    <div className="page">
      <div className="card">
        {/* Header */}
        <header className="header">
          <div>
            <h1>Potato Leaf Disease Detection</h1>
            <p className="sub">
              Ανέβασε φωτογραφία φύλλου πατάτας και πάρε πρόβλεψη (Healthy, Early / Late blight).
            </p>
          </div>

          <div className="actions">
            <button
              className="icon-btn"
              onClick={toggleTheme}
              aria-label="Toggle dark mode"
              title={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
            >
              {theme === "dark" ? "🌙" : "☀️"}
            </button>

            <button
              className="btn primary"
              onClick={onPredict}
              disabled={!file || loading}
            >
              {loading ? "Predicting..." : "Predict"}
            </button>
            <button className="btn ghost" onClick={onReset}>
              Reset
            </button>
          </div>
        </header>

        {/* Dropzone */}
        <section
          className={`dropzone ${drag ? "drag" : ""}`}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onDrop={onDrop}
        >
          <div className="drop-inner">
            {!previewURL ? (
              <>
                <div className="drop-icon">📷</div>
                <div className="drop-title">Σύρε &amp; άφησε μια εικόνα εδώ</div>
                <div className="drop-sub">ή κάνε κλικ για να επιλέξεις αρχείο</div>
                <div style={{ marginTop: 14 }}>
                  <button className="btn" onClick={onPick}>
                    Επιλογή αρχείου
                  </button>
                </div>
              </>
            ) : (
              <img className="preview" src={previewURL} alt="preview" />
            )}
            <input
              ref={inputRef}
              className="hide"
              type="file"
              accept="image/png,image/jpeg"
              onChange={onFileChange}
            />
          </div>
        </section>

        {/* Error */}
        {error && <div className="error">{error}</div>}

        {/* Results */}
        {result && (
          <section className="result">
            <div className="badge">Αποτέλεσμα: {result.label}</div>
            <div className="conf">
              Εμπιστοσύνη: {prettyPercent(result.confidence)}
            </div>

            <div className="result-grid">
              <div className="bars">
                {bars.map(([label, p]) => (
                  <div className="bar-row" key={label}>
                    <div>{label}</div>
                    <div
                      style={{
                        display: "grid",
                        gridTemplateColumns: "1fr auto",
                        gap: 8,
                        alignItems: "center",
                      }}
                    >
                      <div className="bar-track">
                        <div
                          className={`bar-fill ${barClass(label)}`}
                          style={{ width: `${Math.max(1, p * 100)}%` }}
                        />
                      </div>
                      <div>{prettyPercent(p)}</div>
                    </div>
                  </div>
                ))}
              </div>

              {/* SHAP image */}
              {shapImage && (
                <div className="shap-section">
                  <h3>Grad-CAM Explanation</h3>
                  <img
                    src={shapImage}
                    alt="Grad-CAM explanation"
                    className="shap-img"
                  />
                </div>
              )}
            </div>
          </section>
        )}
      </div>
    </div>
  );
}
