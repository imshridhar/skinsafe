import React, { useState, useEffect } from 'react';
import { getScanHistory, deleteScanRecord, BACKEND_URL } from '../services/api';
import type { DiagnosticHistoryItem } from '../types';

interface ScanHistoryTabProps {
  onSelectScan: (item: DiagnosticHistoryItem) => void;
}

export const ScanHistoryTab: React.FC<ScanHistoryTabProps> = ({ onSelectScan }) => {
  const [history, setHistory] = useState<DiagnosticHistoryItem[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterStatus, setFilterStatus] = useState<string>('all');
  const [loading, setLoading] = useState<boolean>(true);
  const [inspectedItem, setInspectedItem] = useState<DiagnosticHistoryItem | null>(null);

  const fetchHistory = async () => {
    setLoading(true);
    try {
      const res = await getScanHistory(searchTerm, filterStatus);
      setHistory(res.scans || []);
    } catch (err) {
      console.error('Failed to load history from backend:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, [searchTerm, filterStatus]);

  const handleDelete = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!window.confirm('Delete this diagnostic scan audit record from database?')) return;
    try {
      await deleteScanRecord(id);
      setHistory((prev) => prev.filter((item) => item.id !== id));
      if (inspectedItem?.id === id) {
        setInspectedItem(null);
      }
    } catch (err) {
      console.error('Failed to delete scan record:', err);
    }
  };

  const handleExportCSV = () => {
    if (history.length === 0) return;
    const headers = ['ID', 'Timestamp', 'Patient ID', 'Age', 'Sex', 'Site', 'Prediction', 'Confidence (%)', 'Status'];
    const rows = history.map((h) => [
      h.id,
      h.timestamp,
      h.patientId,
      h.patientAge || '55',
      h.patientSex || 'unknown',
      h.anatomSite || 'torso',
      `"${h.predictedClass}"`,
      (h.confidence * 100).toFixed(1),
      h.status === 'in_distribution' ? 'In-Distribution' : 'OOD Outlier',
    ]);
    const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map((e) => e.join(','))].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `skinsafe_scan_registry_${new Date().toISOString().slice(0, 10)}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Full Diagnostic Inspection Dossier Modal */}
      {inspectedItem && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            backgroundColor: 'rgba(5, 9, 16, 0.88)',
            backdropFilter: 'blur(16px)',
            zIndex: 1000,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '20px',
          }}
          onClick={() => setInspectedItem(null)}
        >
          <div
            className="glass-panel"
            style={{
              width: '100%',
              maxWidth: '820px',
              maxHeight: '90vh',
              overflowY: 'auto',
              padding: '32px',
              borderRadius: '16px',
              background: 'linear-gradient(180deg, rgba(17, 26, 44, 0.98) 0%, rgba(10, 16, 28, 0.99) 100%)',
              boxShadow: '0 24px 64px rgba(0, 0, 0, 0.85), 0 0 0 1px rgba(0, 210, 211, 0.25)',
              border: '1px solid rgba(0, 210, 211, 0.3)',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Dossier Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '20px', borderBottom: '1px solid rgba(255, 255, 255, 0.08)', paddingBottom: '16px' }}>
              <div>
                <span style={{ fontSize: '11px', fontWeight: 700, color: '#00D2D3', letterSpacing: '0.08em', textTransform: 'uppercase' }}>
                  Clinical Audit Dossier
                </span>
                <h2 style={{ fontSize: '24px', fontWeight: 800, color: '#FFFFFF', marginTop: '2px' }}>
                  Patient {inspectedItem.patientId}
                </h2>
                <p style={{ fontSize: '12px', color: '#94A3B8', marginTop: '2px' }}>
                  Recorded on {inspectedItem.timestamp} · Session ID: <code style={{ color: '#00D2D3' }}>{inspectedItem.id}</code>
                </p>
              </div>
              <button
                onClick={() => setInspectedItem(null)}
                style={{
                  background: 'rgba(255, 255, 255, 0.06)',
                  border: '1px solid rgba(255, 255, 255, 0.12)',
                  color: '#94A3B8',
                  cursor: 'pointer',
                  width: '32px',
                  height: '32px',
                  borderRadius: '8px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                ✕
              </button>
            </div>

            {/* Demographics & Clinical Assessment Bar */}
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))',
                gap: '14px',
                marginBottom: '24px',
                padding: '16px',
                background: 'rgba(8, 12, 20, 0.75)',
                borderRadius: '12px',
                border: '1px solid rgba(255, 255, 255, 0.08)',
              }}
            >
              <div>
                <span style={{ fontSize: '11px', color: '#94A3B8' }}>Age / Sex</span>
                <p style={{ fontSize: '14px', fontWeight: 700, color: '#F8FAFC', marginTop: '2px', textTransform: 'capitalize' }}>
                  {inspectedItem.patientAge || '55'} yrs · {inspectedItem.patientSex || 'unknown'}
                </p>
              </div>
              <div>
                <span style={{ fontSize: '11px', color: '#94A3B8' }}>Anatomical Site</span>
                <p style={{ fontSize: '14px', fontWeight: 700, color: '#F8FAFC', marginTop: '2px', textTransform: 'capitalize' }}>
                  {inspectedItem.anatomSite || 'anterior torso'}
                </p>
              </div>
              <div>
                <span style={{ fontSize: '11px', color: '#94A3B8' }}>Calibrated Confidence</span>
                <p style={{ fontSize: '14px', fontWeight: 800, color: '#00D2D3', marginTop: '2px' }}>
                  {(inspectedItem.confidence * 100).toFixed(1)}%
                </p>
              </div>
              <div>
                <span style={{ fontSize: '11px', color: '#94A3B8' }}>Status</span>
                <div style={{ marginTop: '4px' }}>
                  <span
                    style={{
                      padding: '3px 10px',
                      borderRadius: '10px',
                      fontSize: '11px',
                      fontWeight: 700,
                      background: inspectedItem.status === 'in_distribution' ? 'rgba(16, 185, 129, 0.12)' : 'rgba(244, 63, 94, 0.12)',
                      color: inspectedItem.status === 'in_distribution' ? '#10B981' : '#FDA4AF',
                      border: `1px solid ${inspectedItem.status === 'in_distribution' ? 'rgba(16, 185, 129, 0.3)' : 'rgba(244, 63, 94, 0.3)'}`,
                    }}
                  >
                    {inspectedItem.status === 'in_distribution' ? 'In-Distribution' : 'OOD Outlier'}
                  </span>
                </div>
              </div>
            </div>

            {/* Assessment Callout */}
            <div
              style={{
                padding: '16px 20px',
                borderRadius: '12px',
                marginBottom: '24px',
                background: 'rgba(0, 210, 211, 0.1)',
                border: '1px solid rgba(0, 210, 211, 0.3)',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                flexWrap: 'wrap',
                gap: '12px',
              }}
            >
              <div>
                <span style={{ fontSize: '11px', color: '#94A3B8', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                  AI Pathology Assessment
                </span>
                <h3 style={{ fontSize: '20px', fontWeight: 800, color: '#FFFFFF', marginTop: '2px' }}>
                  {inspectedItem.predictedClass}
                </h3>
              </div>
              <button
                onClick={() => {
                  onSelectScan(inspectedItem);
                  setInspectedItem(null);
                }}
                style={{
                  padding: '9px 18px',
                  borderRadius: '8px',
                  border: 'none',
                  background: 'linear-gradient(135deg, #00D2D3 0%, #0EA5E9 100%)',
                  color: '#060A12',
                  fontWeight: 700,
                  fontSize: '13px',
                  cursor: 'pointer',
                  boxShadow: '0 4px 16px rgba(0, 210, 211, 0.3)',
                }}
              >
                🔬 Open in Diagnostic Scanner
              </button>
            </div>

            {/* Visual Evidence: Original vs Grad-CAM */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
              <div style={{ background: 'rgba(8, 12, 20, 0.85)', padding: '16px', borderRadius: '12px', border: '1px solid rgba(255, 255, 255, 0.08)' }}>
                <span style={{ display: 'block', fontSize: '12px', fontWeight: 600, color: '#94A3B8', marginBottom: '10px' }}>
                  Original Dermoscopy Ingestion
                </span>
                <div style={{ width: '100%', aspectRatio: '1/1', borderRadius: '8px', overflow: 'hidden', background: '#000' }}>
                  {inspectedItem.imageUrl ? (
                    <img
                      src={`${BACKEND_URL}${inspectedItem.imageUrl}`}
                      alt="Original Dermoscopy"
                      style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                      onError={(e) => { e.currentTarget.src = 'https://placehold.co/400x400/0b111e/94a3b8?text=Image+Archived'; }}
                    />
                  ) : (
                    <div style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#64748B' }}>
                      No Image Recorded
                    </div>
                  )}
                </div>
              </div>

              <div style={{ background: 'rgba(8, 12, 20, 0.85)', padding: '16px', borderRadius: '12px', border: '1px solid rgba(255, 255, 255, 0.08)' }}>
                <span style={{ display: 'block', fontSize: '12px', fontWeight: 600, color: '#00D2D3', marginBottom: '10px' }}>
                  Grad-CAM Convolutional Heatmap
                </span>
                <div style={{ width: '100%', aspectRatio: '1/1', borderRadius: '8px', overflow: 'hidden', background: '#000' }}>
                  {inspectedItem.gradcamUrl ? (
                    <img
                      src={`${BACKEND_URL}${inspectedItem.gradcamUrl}`}
                      alt="Grad-CAM Saliency"
                      style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                      onError={(e) => { e.currentTarget.src = 'https://placehold.co/400x400/0b111e/00d2d3?text=GradCAM+Archived'; }}
                    />
                  ) : (
                    <div style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#64748B' }}>
                      No Heatmap Generated
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Main Registry Table */}
      <div className="glass-panel" style={{ padding: '28px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px', flexWrap: 'wrap', gap: '16px' }}>
          <div>
            <h2 style={{ fontSize: '20px', fontWeight: 700, color: '#FFFFFF' }}>
              Clinical Diagnostic Scan Registry & Audit Trail
            </h2>
            <p style={{ fontSize: '13px', color: '#94A3B8', marginTop: '4px' }}>
              Persistently synced with SkinSafe Backend Microservice ({history.length} logged sessions)
            </p>
          </div>

          <div style={{ display: 'flex', gap: '12px', alignItems: 'center', flexWrap: 'wrap' }}>
            <input
              type="text"
              placeholder="Search patient ID, class..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              style={{
                padding: '9px 14px',
                background: 'rgba(8, 12, 20, 0.85)',
                border: '1px solid rgba(255, 255, 255, 0.12)',
                borderRadius: '8px',
                color: '#F8FAFC',
                fontSize: '13px',
                outline: 'none',
                minWidth: '220px',
              }}
            />

            <select
              value={filterStatus}
              onChange={(e) => setFilterStatus(e.target.value)}
              style={{
                padding: '9px 14px',
                background: 'rgba(8, 12, 20, 0.85)',
                border: '1px solid rgba(255, 255, 255, 0.12)',
                borderRadius: '8px',
                color: '#F8FAFC',
                fontSize: '13px',
                outline: 'none',
                cursor: 'pointer',
              }}
            >
              <option value="all">All Records</option>
              <option value="in_distribution">In-Distribution</option>
              <option value="ood">Out-Of-Distribution / Artifact</option>
            </select>

            <button
              onClick={handleExportCSV}
              disabled={history.length === 0}
              style={{
                padding: '9px 16px',
                borderRadius: '8px',
                border: '1px solid rgba(255, 255, 255, 0.15)',
                background: history.length === 0 ? 'rgba(255, 255, 255, 0.04)' : 'rgba(255, 255, 255, 0.08)',
                color: history.length === 0 ? '#64748B' : '#F8FAFC',
                fontWeight: 600,
                fontSize: '13px',
                cursor: history.length === 0 ? 'not-allowed' : 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
              }}
            >
              📥 Export CSV
            </button>
          </div>
        </div>

        {loading ? (
          <div style={{ padding: '40px', textAlign: 'center', color: '#94A3B8' }}>
            Querying backend scan registry...
          </div>
        ) : history.length === 0 ? (
          <div style={{ padding: '48px 24px', textAlign: 'center', color: '#94A3B8' }}>
            <svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="#64748B" strokeWidth="1.5" style={{ margin: '0 auto 12px' }}>
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="8" x2="12" y2="12" />
              <line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
            <p style={{ fontSize: '16px', fontWeight: 600, color: '#F8FAFC' }}>No Diagnostic Records Found</p>
            <p style={{ fontSize: '13px', color: '#94A3B8', marginTop: '6px' }}>
              Upload a dermoscopy image in the Diagnostic Scanner tab to log your first audit record.
            </p>
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.1)', color: '#94A3B8' }}>
                  <th style={{ padding: '14px 12px', textAlign: 'left' }}>Session Time</th>
                  <th style={{ padding: '14px 12px', textAlign: 'left' }}>Patient Code</th>
                  <th style={{ padding: '14px 12px', textAlign: 'left' }}>Demographics</th>
                  <th style={{ padding: '14px 12px', textAlign: 'left' }}>Pathology Assessment</th>
                  <th style={{ padding: '14px 12px', textAlign: 'center' }}>Calibrated Confidence</th>
                  <th style={{ padding: '14px 12px', textAlign: 'center' }}>Status</th>
                  <th style={{ padding: '14px 12px', textAlign: 'right' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {history.map((item) => (
                  <tr
                    key={item.id}
                    style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.04)', cursor: 'pointer', transition: 'background 0.2s' }}
                    onClick={() => setInspectedItem(item)}
                  >
                    <td style={{ padding: '14px 12px', color: '#64748B' }}>{item.timestamp}</td>
                    <td style={{ padding: '14px 12px', fontWeight: 700, color: '#F8FAFC' }}>{item.patientId}</td>
                    <td style={{ padding: '14px 12px', color: '#94A3B8', textTransform: 'capitalize' }}>
                      {item.patientAge || '55'}y · {item.patientSex || 'unknown'} · {item.anatomSite || 'torso'}
                    </td>
                    <td style={{ padding: '14px 12px', fontWeight: 600, color: '#E2E8F0' }}>{item.predictedClass}</td>
                    <td style={{ padding: '14px 12px', textAlign: 'center', fontWeight: 700, color: '#00D2D3' }}>
                      {(item.confidence * 100).toFixed(1)}%
                    </td>
                    <td style={{ padding: '14px 12px', textAlign: 'center' }}>
                      <span
                        style={{
                          padding: '4px 12px',
                          borderRadius: '12px',
                          fontSize: '11px',
                          fontWeight: 700,
                          background: item.status === 'in_distribution' ? 'rgba(16, 185, 129, 0.12)' : 'rgba(244, 63, 94, 0.12)',
                          color: item.status === 'in_distribution' ? '#10B981' : '#FDA4AF',
                          border: `1px solid ${item.status === 'in_distribution' ? 'rgba(16, 185, 129, 0.3)' : 'rgba(244, 63, 94, 0.3)'}`,
                        }}
                      >
                        {item.status === 'in_distribution' ? 'In-Distribution' : 'OOD Outlier'}
                      </span>
                    </td>
                    <td style={{ padding: '14px 12px', textAlign: 'right' }}>
                      <div style={{ display: 'inline-flex', gap: '8px' }}>
                        <button
                          onClick={(e) => { e.stopPropagation(); setInspectedItem(item); }}
                          style={{
                            padding: '6px 14px',
                            borderRadius: '6px',
                            background: 'rgba(0, 210, 211, 0.12)',
                            border: '1px solid rgba(0, 210, 211, 0.35)',
                            color: '#00D2D3',
                            fontSize: '12px',
                            fontWeight: 700,
                            cursor: 'pointer',
                          }}
                        >
                          Inspect Full Record
                        </button>
                        <button
                          onClick={(e) => handleDelete(item.id, e)}
                          title="Delete Record"
                          style={{
                            padding: '6px 10px',
                            borderRadius: '6px',
                            background: 'rgba(244, 63, 94, 0.1)',
                            border: '1px solid rgba(244, 63, 94, 0.25)',
                            color: '#FDA4AF',
                            fontSize: '12px',
                            cursor: 'pointer',
                          }}
                        >
                          ✕
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};
