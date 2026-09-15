import React, { useState, useRef } from 'react';

interface DropzoneProps {
  onImageSelected: (file: File) => void;
  isLoading: boolean;
  isAuthenticated: boolean;
  onRequireAuth: () => void;
}

export const Dropzone: React.FC<DropzoneProps> = ({
  onImageSelected,
  isLoading,
  isAuthenticated,
  onRequireAuth,
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const validateAndPassFile = (file: File) => {
    setErrorMessage(null);

    if (!isAuthenticated) {
      onRequireAuth();
      return;
    }

    // Max 10MB
    if (file.size > 10 * 1024 * 1024) {
      setErrorMessage('Image exceeds maximum clinical file size limit of 10 MB');
      return;
    }

    // Type validation
    if (!['image/jpeg', 'image/png', 'image/jpg'].includes(file.type)) {
      setErrorMessage('Invalid file type. Only JPEG and PNG dermoscopy images are accepted');
      return;
    }

    onImageSelected(file);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (!isAuthenticated) {
      onRequireAuth();
      return;
    }
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndPassFile(e.dataTransfer.files[0]);
    }
  };

  const handleClick = () => {
    if (!isAuthenticated) {
      onRequireAuth();
      return;
    }
    fileInputRef.current?.click();
  };

  // Locked State for Unauthenticated Users
  if (!isAuthenticated) {
    return (
      <div className="w-full">
        <div
          onClick={onRequireAuth}
          style={{
            border: '2px dashed rgba(255, 255, 255, 0.12)',
            borderRadius: '12px',
            padding: '40px 20px',
            textAlign: 'center',
            cursor: 'pointer',
            background: 'rgba(8, 12, 20, 0.75)',
            transition: 'all 0.25s ease',
          }}
          role="button"
          tabIndex={0}
          aria-label="Clinician authentication required to upload dermoscopy images"
        >
          <div style={{ marginBottom: '14px' }}>
            <div
              style={{
                width: '54px',
                height: '54px',
                borderRadius: '12px',
                background: 'rgba(245, 158, 11, 0.12)',
                border: '1px solid rgba(245, 158, 11, 0.3)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                margin: '0 auto',
              }}
            >
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#F59E0B" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
                <path d="M7 11V7a5 5 0 0 1 10 0v4" />
              </svg>
            </div>
          </div>

          <h3 style={{ color: '#F8FAFC', fontSize: '16px', fontWeight: 700, marginBottom: '6px' }}>
            Clinician Authentication Required
          </h3>
          <p style={{ color: '#94A3B8', fontSize: '13px', marginBottom: '18px', maxWidth: '380px', margin: '0 auto 18px' }}>
            Please authenticate with your medical credentials to upload patient dermoscopic scans.
          </p>

          <button
            type="button"
            onClick={(e) => { e.stopPropagation(); onRequireAuth(); }}
            style={{
              padding: '10px 22px',
              borderRadius: '8px',
              background: 'linear-gradient(135deg, #00D2D3 0%, #0EA5E9 100%)',
              border: 'none',
              color: '#060A12',
              fontWeight: 800,
              fontSize: '13px',
              cursor: 'pointer',
              boxShadow: '0 4px 16px rgba(0, 210, 211, 0.3)',
              display: 'inline-flex',
              alignItems: 'center',
              gap: '8px',
            }}
          >
            <span>🔐 Clinician Sign In to Access Scanner</span>
          </button>
        </div>
      </div>
    );
  }

  // Unlocked Dropzone
  return (
    <div className="w-full">
      <div
        onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={handleDrop}
        onClick={handleClick}
        style={{
          border: isDragOver ? '2px dashed #00D2D3' : '2px dashed rgba(255, 255, 255, 0.14)',
          borderRadius: '12px',
          padding: '44px 20px',
          textAlign: 'center',
          cursor: isLoading ? 'not-allowed' : 'pointer',
          background: isDragOver ? 'rgba(0, 210, 211, 0.06)' : 'rgba(8, 12, 20, 0.65)',
          transition: 'all 0.25s ease',
          boxShadow: isDragOver ? '0 0 30px rgba(0, 210, 211, 0.15)' : 'none',
        }}
        role="button"
        tabIndex={0}
        aria-label="Upload dermoscopy image for AI diagnostic analysis"
        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') handleClick(); }}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/jpeg,image/png,image/jpg"
          className="hidden"
          style={{ display: 'none' }}
          onChange={(e) => {
            if (e.target.files && e.target.files.length > 0) {
              validateAndPassFile(e.target.files[0]);
            }
          }}
          disabled={isLoading}
        />

        <div style={{ marginBottom: '14px' }}>
          <div
            style={{
              width: '54px',
              height: '54px',
              borderRadius: '12px',
              background: 'linear-gradient(135deg, rgba(0, 210, 211, 0.15) 0%, rgba(14, 165, 233, 0.15) 100%)',
              border: '1px solid rgba(0, 210, 211, 0.3)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              margin: '0 auto',
            }}
          >
            <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#00D2D3" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
          </div>
        </div>

        <h3 style={{ color: '#F8FAFC', fontSize: '16px', fontWeight: 700, marginBottom: '6px' }}>
          {isLoading ? 'Running Deep Diagnostic Inference...' : 'Drag & Drop Dermoscopic Lesion'}
        </h3>
        <p style={{ color: '#94A3B8', fontSize: '13px', marginBottom: '14px' }}>
          or click to browse local hospital imaging filesystem
        </p>

        <div style={{ display: 'inline-flex', gap: '10px', fontSize: '11px', color: '#64748B', background: 'rgba(255, 255, 255, 0.04)', padding: '4px 12px', borderRadius: '20px', border: '1px solid rgba(255, 255, 255, 0.06)' }}>
          <span>JPEG / PNG</span>
          <span>•</span>
          <span>Max 10 MB</span>
          <span>•</span>
          <span>Standard Dermoscopy</span>
        </div>
      </div>

      {errorMessage && (
        <div style={{ marginTop: '12px', padding: '12px', background: 'rgba(244, 63, 94, 0.12)', border: '1px solid rgba(244, 63, 94, 0.3)', borderRadius: '8px', color: '#FDA4AF', fontSize: '13px' }}>
          {errorMessage}
        </div>
      )}
    </div>
  );
};
