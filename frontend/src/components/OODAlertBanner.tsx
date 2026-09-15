import React from 'react';
import type { DiagnosticStatus } from '../types';

interface OODAlertBannerProps {
  status: DiagnosticStatus;
  energyScore?: number;
}

export const OODAlertBanner: React.FC<OODAlertBannerProps> = ({
  status,
  energyScore,
}) => {
  if (status !== 'ood') {
    return null;
  }

  return (
    <div
      style={{
        padding: '16px 20px',
        background: 'rgba(244, 63, 94, 0.12)',
        border: '1px solid rgba(244, 63, 94, 0.4)',
        borderRadius: '12px',
        display: 'flex',
        alignItems: 'center',
        gap: '14px',
        boxShadow: '0 4px 20px rgba(0, 0, 0, 0.3)',
      }}
    >
      <span style={{ fontSize: '22px' }}>🚨</span>
      <div style={{ flex: 1 }}>
        <h4 style={{ color: '#FDA4AF', fontSize: '14px', fontWeight: 700 }}>
          Out-Of-Distribution (OOD) / Severe Optical Artifact Flagged
        </h4>
        <p style={{ color: '#CBD5E1', fontSize: '13px', marginTop: '2px' }}>
          Free Energy threshold exceeded {energyScore !== undefined ? `(E = ${energyScore.toFixed(2)})` : ''}. Lesion features deviate significantly from known training distribution.
        </p>
      </div>
    </div>
  );
};

