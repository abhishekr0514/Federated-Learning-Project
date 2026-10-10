import React, { useState, useEffect } from 'react';
import { Shield, Activity, Server, Database, RefreshCw, AlertCircle, Cpu } from 'lucide-react';

export default function LegacyDashboardApp() {
  const [healthData, setHealthData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchHealth = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/health');
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setHealthData(data);
    } catch (err) {
      console.error('Failed to fetch backend health:', err);
      setError(err.message || 'Unable to connect to backend service');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHealth();
  }, []);

  return (
    <div className="container">
      <header className="header">
        <div className="title-group">
          <h1>MedShield-FL Console (Legacy Health View)</h1>
          <p>Privacy Masked Federated Learning for Heart Disease Diagnosis</p>
        </div>
        <div>
          {loading ? (
            <span className="badge">Checking Backend...</span>
          ) : error ? (
            <span className="badge offline">
              <AlertCircle size={14} /> Backend Disconnected
            </span>
          ) : (
            <span className="badge success">
              <span className="status-indicator"></span> Backend Operational
            </span>
          )}
        </div>
      </header>

      <div className="grid">
        {/* Card 1: Backend System Health */}
        <div className="card">
          <div className="card-header">
            <div className="card-icon">
              <Server size={20} />
            </div>
            <div className="card-title">Backend API Health</div>
          </div>
          <div className="info-list">
            <div className="info-item">
              <span className="info-label">Status</span>
              <span className="info-value">{loading ? 'Checking...' : error ? 'Offline' : healthData?.status}</span>
            </div>
            <div className="info-item">
              <span className="info-label">Service Name</span>
              <span className="info-value">{healthData?.service || 'medshield-fl-backend'}</span>
            </div>
            <div className="info-item">
              <span className="info-label">Version</span>
              <span className="info-value">{healthData?.version || '0.1.0'}</span>
            </div>
            <div className="info-item">
              <span className="info-label">Environment</span>
              <span className="info-value">{healthData?.environment || 'development'}</span>
            </div>
          </div>
          <button className="btn" onClick={fetchHealth} style={{ marginTop: '1.25rem', width: '100%', justifyContent: 'center' }}>
            <RefreshCw size={14} /> Refresh Health Status
          </button>
        </div>

        {/* Card 2: Neural Model Contracts */}
        <div className="card">
          <div className="card-header">
            <div className="card-icon">
              <Cpu size={20} />
            </div>
            <div className="card-title">Neural Sub-Networks</div>
          </div>
          <div className="info-list">
            <div className="info-item">
              <span className="info-label">ECG Encoder</span>
              <span className="info-value">1D CNN + BiLSTM (128d)</span>
            </div>
            <div className="info-item">
              <span className="info-label">Text Encoder</span>
              <span className="info-value">BioClinicalBERT (128d)</span>
            </div>
            <div className="info-item">
              <span className="info-label">Tabular Encoder</span>
              <span className="info-value">Neural Tabular (64d)</span>
            </div>
            <div className="info-item">
              <span className="info-label">Multimodal Fusion</span>
              <span className="info-value">GAT Layers (384d ➔ Logits)</span>
            </div>
          </div>
        </div>

        {/* Card 3: Privacy & Security Rules */}
        <div className="card">
          <div className="card-header">
            <div className="card-icon">
              <Shield size={20} />
            </div>
            <div className="card-title">Privacy Boundaries</div>
          </div>
          <div className="info-list">
            <div className="info-item">
              <span className="info-label">Raw ECG / Text</span>
              <span className="info-value">Local Hospital Only</span>
            </div>
            <div className="info-item">
              <span className="info-label">Central DB Storage</span>
              <span className="info-value">Zero Raw Text / PII</span>
            </div>
            <div className="info-item">
              <span className="info-label">Federated Engine</span>
              <span className="info-value">Flower Framework</span>
            </div>
            <div className="info-item">
              <span className="info-label">Differential Privacy</span>
              <span className="info-value">L2 Clip + Gaussian Noise</span>
            </div>
          </div>
        </div>
      </div>

      {/* Audit & Scientific Notice Banner */}
      <div className="notice-banner">
        <h3><Activity size={16} style={{ display: 'inline', marginRight: '6px', verticalAlign: 'middle' }} /> Development Audit & Proxy Task Notice</h3>
        <p>
          Prompt 01 repository foundation and safe checkout restoration complete. The backend <code>GET /health</code> endpoint is fully operational.
          The PTB-XL binary target currently evaluates normal sinus rhythm (SNOMED 426783006) vs pathological superclasses as a proxy task. Full multimodal GAT inference endpoints will be enabled in future phases after validation without metric fabrication.
        </p>
      </div>
    </div>
  );
}
