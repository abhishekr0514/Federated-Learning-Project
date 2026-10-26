import React, { useState, useRef, useEffect, useMemo } from 'react';
import { Activity, Eye, ZoomIn, Layers, Maximize2 } from 'lucide-react';

const LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'];

export function EcgWaveformCanvas({ leads, title = '12-Lead Electrocardiogram' }) {
  const [selectedLead, setSelectedLead] = useState(1); // Default to Lead II (index 1)
  const [viewMode, setViewMode] = useState('single'); // 'single' | 'all12'
  const [hoverData, setHoverData] = useState(null);
  const canvasRef = useRef(null);

  const isValidLeads = useMemo(() => {
    return Array.isArray(leads) && leads.length === 12 && leads.every((l) => Array.isArray(l) && l.length === 1000);
  }, [leads]);

  // Render on canvas when leads or selected lead changes
  useEffect(() => {
    if (!isValidLeads || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    // High DPI scaling
    const dpr = window.devicePixelRatio || 1;
    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // 1. Draw ECG Grid Background
    const gridSize = 16;
    ctx.strokeStyle = 'rgba(13, 148, 136, 0.08)';
    ctx.lineWidth = 1;
    for (let x = 0; x < width; x += gridSize) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = 0; y < height; y += gridSize) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Heavy grid lines every 5 subdivisions (standard ECG paper 0.2s / 0.5mV)
    ctx.strokeStyle = 'rgba(13, 148, 136, 0.18)';
    ctx.lineWidth = 1.2;
    for (let x = 0; x < width; x += gridSize * 5) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = 0; y < height; y += gridSize * 5) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // 2. Draw Waveforms
    if (viewMode === 'single') {
      const leadData = leads[selectedLead];
      const leadName = LEAD_NAMES[selectedLead] || `Lead ${selectedLead + 1}`;

      // Zero volt baseline center
      const centerY = height / 2;
      const mvScale = (height * 0.38) / 1.5; // ~1.5 mV fits comfortably

      // Glow effect
      ctx.shadowBlur = 6;
      ctx.shadowColor = '#06b6d4';
      ctx.strokeStyle = '#22d3ee';
      ctx.lineWidth = 2.0;
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';

      ctx.beginPath();
      for (let i = 0; i < leadData.length; i++) {
        const x = (i / (leadData.length - 1)) * width;
        const y = centerY - leadData[i] * mvScale;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Reset shadow
      ctx.shadowBlur = 0;

      // Draw Baseline reference line
      ctx.strokeStyle = 'rgba(148, 163, 184, 0.25)';
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(0, centerY);
      ctx.lineTo(width, centerY);
      ctx.stroke();
      ctx.setLineDash([]);

      // Draw Calibration pulse on the left (1mV standard)
      ctx.strokeStyle = '#0ea5e9';
      ctx.lineWidth = 1.5;
      const calX = 14;
      const calW = 12;
      const calH = 1.0 * mvScale;
      ctx.beginPath();
      ctx.moveTo(calX, centerY);
      ctx.lineTo(calX + 4, centerY);
      ctx.lineTo(calX + 4, centerY - calH);
      ctx.lineTo(calX + 4 + calW, centerY - calH);
      ctx.lineTo(calX + 4 + calW, centerY);
      ctx.lineTo(calX + 4 + calW + 4, centerY);
      ctx.stroke();

      // Annotation text
      ctx.fillStyle = '#94a3b8';
      ctx.font = '10px Inter, monospace';
      ctx.fillText('1.0 mV Cal', calX, centerY - calH - 4);
      ctx.fillText(`${leadName} — 25mm/s, 10mm/mV, 100Hz (1000 samples / 10s)`, 110, 20);
    } else {
      // 12-lead Multi-channel stacked view (12 parallel horizontal strips)
      const stripHeight = height / 12;

      leads.forEach((leadData, idx) => {
        const centerY = (idx + 0.5) * stripHeight;
        const mvScale = stripHeight * 0.4;

        // Label
        ctx.fillStyle = '#38bdf8';
        ctx.font = 'bold 10px Inter, sans-serif';
        ctx.fillText(LEAD_NAMES[idx], 8, centerY - 2);

        // Waveform
        ctx.strokeStyle = idx === selectedLead ? '#38bdf8' : '#0d9488';
        ctx.lineWidth = idx === selectedLead ? 1.8 : 1.2;
        ctx.beginPath();

        for (let i = 0; i < leadData.length; i++) {
          const x = 36 + (i / (leadData.length - 1)) * (width - 44);
          const y = centerY - leadData[i] * mvScale;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Strip divider line
        if (idx < 11) {
          ctx.strokeStyle = 'rgba(55, 65, 81, 0.4)';
          ctx.lineWidth = 0.8;
          ctx.beginPath();
          ctx.moveTo(0, (idx + 1) * stripHeight);
          ctx.lineTo(width, (idx + 1) * stripHeight);
          ctx.stroke();
        }
      });
    }
  }, [leads, selectedLead, viewMode, isValidLeads]);

  const handleMouseMove = (e) => {
    if (!canvasRef.current || !isValidLeads || viewMode !== 'single') return;
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const normX = Math.max(0, Math.min(1, x / rect.width));
    const sampleIdx = Math.floor(normX * 999);
    const val = leads[selectedLead][sampleIdx];
    const timeSec = (sampleIdx / 100).toFixed(2);
    setHoverData({ sampleIdx, timeSec, valueMv: val });
  };

  const handleMouseLeave = () => {
    setHoverData(null);
  };

  if (!isValidLeads) {
    return (
      <div className="ecg-canvas-empty" style={{
        padding: '2rem',
        textAlign: 'center',
        background: 'var(--bg-secondary)',
        border: '1px dashed var(--border-color)',
        borderRadius: 'var(--radius-md)',
        color: 'var(--text-muted)'
      }}>
        <Activity size={32} style={{ margin: '0 auto 0.75rem', opacity: 0.5 }} />
        <p>No valid 12-lead (12 x 1000) ECG signal data loaded.</p>
        <p style={{ fontSize: '0.8125rem', marginTop: '4px' }}>
          Upload a supported CSV / JSON waveform file or select a PTB-XL benchmark sample below.
        </p>
      </div>
    );
  }

  return (
    <div className="ecg-waveform-container" style={{
      background: '#090d16',
      border: '1px solid #1e293b',
      borderRadius: 'var(--radius-lg)',
      padding: '1rem',
      position: 'relative'
    }}>
      <div className="ecg-toolbar" style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '0.75rem',
        flexWrap: 'wrap',
        gap: '0.5rem'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Activity size={16} className="text-accent" />
          <strong style={{ fontSize: '0.875rem', color: '#f8fafc' }}>{title}</strong>
          <span className="badge" style={{ fontSize: '0.7rem', padding: '0.15rem 0.5rem', background: 'rgba(6, 182, 212, 0.15)', color: '#22d3ee', border: '1px solid rgba(6, 182, 212, 0.3)' }}>
            12 Leads • 1000 Samples • 10.0s @ 100Hz
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          {/* Mode switch */}
          <div style={{ display: 'inline-flex', background: '#1e293b', borderRadius: 'var(--radius-sm)', padding: '2px' }}>
            <button
              type="button"
              className={`btn btn-sm ${viewMode === 'single' ? 'btn-primary' : ''}`}
              style={{ padding: '0.2rem 0.6rem', fontSize: '0.75rem', border: 'none' }}
              onClick={() => setViewMode('single')}
            >
              <Eye size={12} style={{ marginRight: '4px' }} /> Focused Lead
            </button>
            <button
              type="button"
              className={`btn btn-sm ${viewMode === 'all12' ? 'btn-primary' : ''}`}
              style={{ padding: '0.2rem 0.6rem', fontSize: '0.75rem', border: 'none' }}
              onClick={() => setViewMode('all12')}
            >
              <Layers size={12} style={{ marginRight: '4px' }} /> 12-Channel Strip
            </button>
          </div>
        </div>
      </div>

      {/* Single lead pills */}
      {viewMode === 'single' && (
        <div className="lead-selector-bar" style={{
          display: 'flex',
          gap: '4px',
          overflowX: 'auto',
          paddingBottom: '6px',
          marginBottom: '6px'
        }}>
          {LEAD_NAMES.map((name, idx) => (
            <button
              key={name}
              type="button"
              onClick={() => setSelectedLead(idx)}
              style={{
                padding: '3px 9px',
                fontSize: '0.75rem',
                borderRadius: '4px',
                border: selectedLead === idx ? '1px solid #06b6d4' : '1px solid #334155',
                background: selectedLead === idx ? 'rgba(6, 182, 212, 0.2)' : '#0f172a',
                color: selectedLead === idx ? '#38bdf8' : '#94a3b8',
                fontWeight: selectedLead === idx ? 600 : 400,
                cursor: 'pointer'
              }}
            >
              {name}
            </button>
          ))}
        </div>
      )}

      {/* Canvas */}
      <div style={{ position: 'relative', overflow: 'hidden', borderRadius: 'var(--radius-md)' }}>
        <canvas
          ref={canvasRef}
          width={840}
          height={viewMode === 'single' ? 240 : 360}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
          style={{
            width: '100%',
            height: viewMode === 'single' ? '240px' : '360px',
            display: 'block',
            cursor: viewMode === 'single' ? 'crosshair' : 'default',
            background: '#040711'
          }}
        />

        {/* Hover telemetry tooltip */}
        {hoverData && viewMode === 'single' && (
          <div style={{
            position: 'absolute',
            top: '12px',
            right: '12px',
            background: 'rgba(15, 23, 42, 0.92)',
            border: '1px solid #334155',
            borderRadius: '6px',
            padding: '4px 8px',
            fontSize: '0.75rem',
            color: '#f8fafc',
            fontFamily: 'monospace',
            pointerEvents: 'none',
            boxShadow: '0 4px 12px rgba(0,0,0,0.5)'
          }}>
            <span style={{ color: '#38bdf8' }}>{LEAD_NAMES[selectedLead]}</span>: {hoverData.valueMv.toFixed(3)} mV | t: {hoverData.timeSec}s [sample #{hoverData.sampleIdx}]
          </div>
        )}
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '6px', fontSize: '0.7rem', color: '#64748b' }}>
        <span>Scale: 25 mm/s, 10 mm/mV</span>
        <span>Lead Amplitude Window: [-1.5 mV, +1.5 mV]</span>
        <span>Duration: 10.0 seconds</span>
      </div>
    </div>
  );
}
