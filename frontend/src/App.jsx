import { useState, useEffect } from "react";
import "./App.css";
import PredictionForm from "./components/PredictionForm";
import PredictionResult from "./components/PredictionResult";
import { fetchMetadata, fetchPrediction } from "./api";

function App() {
  const [metadata, setMetadata] = useState({});
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [metaLoading, setMetaLoading] = useState(true);

  useEffect(() => {
    fetchMetadata()
      .then((data) => {
        setMetadata(data);
        setMetaLoading(false);
      })
      .catch(() => {
        setError("Unable to connect to the API server. Please make sure it is running.");
        setMetaLoading(false);
      });
  }, []);

  const handleClear = () => {
    setResult(null);
    setError(null);
  };

  const handlePredict = async (formData) => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const prediction = await fetchPrediction(formData);
      setResult(prediction);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-wrapper">
      <header className="header">
        <div className="header-bg-shapes">
          <div className="header-shape header-shape-1" />
          <div className="header-shape header-shape-2" />
          <div className="header-shape header-shape-3" />
        </div>
        <div className="header-inner">
          <div className="header-logo">
            <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z" />
            </svg>
          </div>
          <h1 className="header-title">
            Flood Risk
            <span className="header-title-accent"> Classification</span>
          </h1>
          <p className="header-desc">Predict flood likelihood for Sri Lankan districts using environmental indicators and machine learning</p>
          <div className="header-tags">
            <span className="header-tag">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M22 12h-4l-3 9L9 3l-3 9H2" /></svg>
              ML-Powered
            </span>
            <span className="header-tag">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><circle cx="12" cy="12" r="10" /><path d="M2 12h20" /><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" /></svg>
              Sri Lanka
            </span>
            <span className="header-tag">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M12 20V10" /><path d="M18 20V4" /><path d="M6 20v-4" /></svg>
              Real-Time Analysis
            </span>
          </div>
        </div>
      </header>

      <main className="main-content">
        {error && (
          <div className="error-banner">
            <div className="error-icon">!</div>
            <span>{error}</span>
          </div>
        )}

        {metaLoading ? (
          <div className="loading-container">
            <div className="spinner" />
            <p>Connecting to prediction API...</p>
          </div>
        ) : (
          <>
            <div className="content-grid">
              <section className="form-section">
                <PredictionForm
                  metadata={metadata}
                  onSubmit={handlePredict}
                  onClear={handleClear}
                  loading={loading}
                />
              </section>
              <section className="result-section">
                <PredictionResult result={result} loading={loading} />
              </section>
            </div>

          </>
        )}
      </main>

      <footer className="footer">
        <p>Flood Risk Classification &middot; Academic ML Project &middot; Sri Lanka</p>
      </footer>
    </div>
  );
}

export default App;
