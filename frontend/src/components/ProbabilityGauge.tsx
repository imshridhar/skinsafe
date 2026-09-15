import React from 'react';

interface ProbabilityGaugeProps {
  probabilities: Record<string, number>;
  predictedClass: string;
}

const CLASS_DESCRIPTIONS: Record<string, { name: string; risk: 'Malignant' | 'Pre-cancer' | 'Benign' | 'Artifact' }> = {
  MEL: { name: 'Melanoma', risk: 'Malignant' },
  NV: { name: 'Melanocytic Nevus', risk: 'Benign' },
  BCC: { name: 'Basal Cell Carcinoma', risk: 'Malignant' },
  AK: { name: 'Actinic Keratosis', risk: 'Pre-cancer' },
  BKL: { name: 'Benign Keratosis', risk: 'Benign' },
  DF: { name: 'Dermatofibroma', risk: 'Benign' },
  VASC: { name: 'Vascular Lesion', risk: 'Benign' },
  SCC: { name: 'Squamous Cell Carcinoma', risk: 'Malignant' },
  UNK: { name: 'Outlier / Artifact', risk: 'Artifact' },
};

export const ProbabilityGauge: React.FC<ProbabilityGaugeProps> = ({
  probabilities,
  predictedClass,
}) => {
  // Sort classes by probability descending
  const sortedClasses = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);

  return (
    <div className="glass-panel" style={{ padding: '24px', width: '100%' }}>
      <h3 style={{ fontSize: '15px', fontWeight: 700, color: '#FFFFFF', marginBottom: '18px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <span>Calibrated Multi-Class Distribution</span>
        <span style={{ fontSize: '11px', color: '#00D2D3', fontWeight: 600, background: 'rgba(0, 210, 211, 0.12)', padding: '2px 8px', borderRadius: '4px' }}>
          T* = 1.2014 Scaled
        </span>
      </h3>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
        {sortedClasses.map(([code, prob]) => {
          const info = CLASS_DESCRIPTIONS[code] || { name: code, risk: 'Benign' };
          const percentage = (prob * 100).toFixed(1);
          const isSelected = code === predictedClass;
          const isMalignant = info.risk === 'Malignant';
          const isPreCancer = info.risk === 'Pre-cancer';

          let riskColor = '#10B981';
          if (isMalignant) riskColor = '#F43F5E';
          else if (isPreCancer) riskColor = '#F59E0B';

          return (
            <div key={code} style={{ display: 'flex', flexDirection: 'column', gap: '5px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '13px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span
                    style={{
                      fontSize: '11px',
                      fontWeight: 800,
                      padding: '2px 6px',
                      borderRadius: '4px',
                      background: `${riskColor}22`,
                      color: riskColor,
                      border: `1px solid ${riskColor}35`,
                    }}
                  >
                    {code}
                  </span>
                  <span style={{ color: isSelected ? '#FFFFFF' : '#CBD5E1', fontWeight: isSelected ? 700 : 500 }}>
                    {info.name}
                  </span>
                </div>

                <span style={{ color: isSelected ? '#00D2D3' : '#94A3B8', fontWeight: isSelected ? 800 : 500, fontFamily: 'monospace', fontSize: '13px' }}>
                  {percentage}%
                </span>
              </div>

              <div style={{ width: '100%', height: '6px', background: 'rgba(255, 255, 255, 0.06)', borderRadius: '3px', overflow: 'hidden' }}>
                <div
                  style={{
                    width: `${percentage}%`,
                    height: '100%',
                    background: isMalignant
                      ? 'linear-gradient(90deg, #F43F5E 0%, #FB7185 100%)'
                      : isPreCancer
                      ? 'linear-gradient(90deg, #F59E0B 0%, #FBBF24 100%)'
                      : 'linear-gradient(90deg, #00D2D3 0%, #0EA5E9 100%)',
                    borderRadius: '3px',
                    transition: 'width 0.6s cubic-bezier(0.16, 1, 0.3, 1)',
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
