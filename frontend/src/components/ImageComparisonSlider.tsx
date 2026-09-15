import React, { useState } from 'react';

interface ImageComparisonSliderProps {
  originalUrl: string;
  gradcamUrl: string;
  altText?: string;
}

export const ImageComparisonSlider: React.FC<ImageComparisonSliderProps> = ({
  originalUrl,
  gradcamUrl,
  altText = 'Dermoscopy image comparison'
}) => {
  const [sliderPos, setSliderPos] = useState(50);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowLeft') {
      setSliderPos((prev) => Math.max(0, prev - 5));
    } else if (e.key === 'ArrowRight') {
      setSliderPos((prev) => Math.min(100, prev + 5));
    }
  };

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '12px',
        width: '100%',
        maxWidth: '520px',
        margin: '0 auto',
      }}
    >
      <div
        tabIndex={0}
        role="slider"
        aria-label="Image comparison slider between original dermoscopy and Grad-CAM saliency"
        aria-valuenow={sliderPos}
        aria-valuemin={0}
        aria-valuemax={100}
        onKeyDown={handleKeyDown}
        style={{
          position: 'relative',
          width: '100%',
          aspectRatio: '1/1',
          borderRadius: 'var(--radius-md)',
          overflow: 'hidden',
          border: '1px solid var(--border-subtle)',
          boxShadow: 'var(--shadow-glass)',
          outline: 'none',
        }}
      >
        {/* Original Image (Background) */}
        <img
          src={originalUrl}
          alt={`Original dermoscopy ${altText}`}
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            objectFit: 'cover',
          }}
        />

        {/* Grad-CAM Saliency Overlay (Clipped) */}
        <div
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            clipPath: `polygon(0 0, ${sliderPos}% 0, ${sliderPos}% 100%, 0 100%)`,
          }}
        >
          <img
            src={gradcamUrl}
            alt={`Grad-CAM explanation ${altText}`}
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: '100%',
              objectFit: 'cover',
            }}
          />
        </div>

        {/* Divider Line */}
        <div
          style={{
            position: 'absolute',
            top: 0,
            bottom: 0,
            left: `${sliderPos}%`,
            width: '2px',
            background: 'var(--accent-cyan)',
            boxShadow: '0 0 8px rgba(86, 216, 232, 0.8)',
            cursor: 'ew-resize',
          }}
        >
          <div
            style={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              width: '28px',
              height: '28px',
              background: 'var(--surface-solid)',
              border: '2px solid var(--accent-cyan)',
              borderRadius: '50%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '10px',
              color: 'var(--accent-cyan)',
            }}
          >
            ↔
          </div>
        </div>

        {/* Floating Labels */}
        <div
          style={{
            position: 'absolute',
            top: '12px',
            left: '12px',
            padding: '4px 8px',
            background: 'rgba(7, 17, 31, 0.75)',
            backdropFilter: 'blur(4px)',
            borderRadius: 'var(--radius-sm)',
            fontSize: '12px',
            color: 'var(--accent-cyan)',
            border: '1px solid var(--border-subtle)',
          }}
        >
          Grad-CAM Saliency
        </div>

        <div
          style={{
            position: 'absolute',
            top: '12px',
            right: '12px',
            padding: '4px 8px',
            background: 'rgba(7, 17, 31, 0.75)',
            backdropFilter: 'blur(4px)',
            borderRadius: 'var(--radius-sm)',
            fontSize: '12px',
            color: 'var(--text-primary)',
            border: '1px solid var(--border-subtle)',
          }}
        >
          Original Image
        </div>
      </div>

      {/* Range Control Slider */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', padding: '0 8px' }}>
        <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Grad-CAM</span>
        <input
          type="range"
          min="0"
          max="100"
          value={sliderPos}
          onChange={(e) => setSliderPos(Number(e.target.value))}
          style={{
            flex: 1,
            accentColor: 'var(--accent-cyan)',
            cursor: 'pointer',
          }}
          aria-label="Comparison position slider"
        />
        <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Original</span>
      </div>
    </div>
  );
};
