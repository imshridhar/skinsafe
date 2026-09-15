import React, { useState } from 'react';
import { signIn, signUp } from '../services/api';

interface AuthModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: (email: string, name: string, role: string) => void;
}

export const AuthModal: React.FC<AuthModalProps> = ({ isOpen, onClose, onSuccess }) => {
  const [isSignIn, setIsSignIn] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [name, setName] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!isOpen) return null;

  const resetForm = () => {
    setEmail('');
    setPassword('');
    setConfirmPassword('');
    setName('');
    setError(null);
    setShowPassword(false);
    setShowConfirmPassword(false);
  };

  const handleClose = () => {
    resetForm();
    onClose();
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      if (isSignIn) {
        const res = await signIn(email, password);
        onSuccess(res.email, res.name || 'Dr. Clinician', res.role || 'Chief Dermatologist');
        resetForm();
        onClose();
      } else {
        if (password !== confirmPassword) {
          setError('Passwords do not match');
          setLoading(false);
          return;
        }
        const res = await signUp(email, password, confirmPassword, name || 'Dr. Clinician');
        onSuccess(res.email, res.name || name, res.role || 'Dermatology Researcher');
        resetForm();
        onClose();
      }
    } catch (err: any) {
      if (err.code === 'ERR_NETWORK' || !err.response) {
        setError('⚠️ Backend Offline: Cannot connect to SkinSafe API on http://127.0.0.1:5000. Start the server using: python run_server.py');
      } else {
        setError(err?.response?.data?.error || err.message || 'Authentication rejected by security gate.');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        backgroundColor: 'rgba(4, 7, 13, 0.90)',
        backdropFilter: 'blur(16px)',
        zIndex: 1000,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '20px',
      }}
      onClick={handleClose}
    >
      <div
        className="glass-panel"
        style={{
          width: '100%',
          maxWidth: '460px',
          padding: '36px',
          borderRadius: '18px',
          background: 'linear-gradient(180deg, #0F172A 0%, #090E1A 100%)',
          boxShadow: '0 24px 64px rgba(0, 0, 0, 0.85), 0 0 0 1px rgba(0, 210, 211, 0.25)',
          border: '1px solid rgba(0, 210, 211, 0.25)',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Modal Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
          <div>
            <h2 style={{ fontSize: '22px', fontWeight: 800, color: '#FFFFFF', letterSpacing: '-0.02em' }}>
              {isSignIn ? 'Clinician Authentication' : 'Create Researcher Account'}
            </h2>
          </div>
          <button
            onClick={handleClose}
            style={{
              background: 'rgba(255, 255, 255, 0.06)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              color: '#94A3B8',
              cursor: 'pointer',
              fontSize: '14px',
              width: '32px',
              height: '32px',
              borderRadius: '8px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              transition: 'all 0.2s',
            }}
          >
            ✕
          </button>
        </div>

        {/* Segmented Tab Switcher */}
        <div
          style={{
            display: 'flex',
            background: '#060A12',
            borderRadius: '10px',
            padding: '4px',
            marginBottom: '24px',
            border: '1px solid rgba(255, 255, 255, 0.08)',
          }}
        >
          <button
            type="button"
            onClick={() => {
              if (!isSignIn) {
                setIsSignIn(true);
                resetForm();
              }
            }}
            style={{
              flex: 1,
              padding: '10px 14px',
              borderRadius: '8px',
              border: 'none',
              background: isSignIn ? 'linear-gradient(135deg, #00D2D3 0%, #0EA5E9 100%)' : 'transparent',
              color: isSignIn ? '#050910' : '#94A3B8',
              fontWeight: 700,
              fontSize: '13px',
              cursor: 'pointer',
              transition: 'all 0.2s',
            }}
          >
            Sign In
          </button>
          <button
            type="button"
            onClick={() => {
              if (isSignIn) {
                setIsSignIn(false);
                resetForm();
              }
            }}
            style={{
              flex: 1,
              padding: '10px 14px',
              borderRadius: '8px',
              border: 'none',
              background: !isSignIn ? 'linear-gradient(135deg, #00D2D3 0%, #0EA5E9 100%)' : 'transparent',
              color: !isSignIn ? '#050910' : '#94A3B8',
              fontWeight: 700,
              fontSize: '13px',
              cursor: 'pointer',
              transition: 'all 0.2s',
            }}
          >
            Register
          </button>
        </div>

        {/* Error Alert Box */}
        {error && (
          <div
            style={{
              padding: '12px 16px',
              background: 'rgba(244, 63, 94, 0.12)',
              border: '1px solid rgba(244, 63, 94, 0.35)',
              borderRadius: '10px',
              color: '#FDA4AF',
              fontSize: '13px',
              lineHeight: 1.5,
              marginBottom: '18px',
            }}
          >
            {error}
          </div>
        )}

        {/* Auth Form */}
        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '18px' }} autoComplete="off">
          {!isSignIn && (
            <div>
              <label style={{ display: 'block', fontSize: '12px', fontWeight: 600, color: '#94A3B8', marginBottom: '6px' }}>
                Full Name & Medical Title
              </label>
              <div className="auth-input-container">
                <input
                  type="text"
                  name="fullname_field"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="Dr. Sarah Jenkins, MD"
                  required
                  className="auth-input-field"
                />
                <div className="auth-input-icon">
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
                    <circle cx="12" cy="7" r="4" />
                  </svg>
                </div>
              </div>
            </div>
          )}

          <div>
            <label style={{ display: 'block', fontSize: '12px', fontWeight: 600, color: '#94A3B8', marginBottom: '6px' }}>
              Institutional Email Address
            </label>
            <div className="auth-input-container">
              <input
                type="email"
                name="institution_email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="doctor@skinsafe.ai"
                required
                autoComplete="off"
                data-form-type="other"
                data-lpignore="true"
                data-1p-ignore="true"
                className="auth-input-field"
              />
              <div className="auth-input-icon">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z" />
                  <polyline points="22,6 12,13 2,6" />
                </svg>
              </div>
            </div>
          </div>

          <div>
            <label style={{ display: 'block', fontSize: '12px', fontWeight: 600, color: '#94A3B8', marginBottom: '6px' }}>
              Password
            </label>
            <div className="auth-input-container">
              <input
                type={showPassword ? 'text' : 'password'}
                name="current_security_key"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Enter password"
                required
                autoComplete="new-password"
                className="auth-input-field"
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                title={showPassword ? 'Hide password' : 'Show password'}
                style={{
                  position: 'absolute',
                  right: '12px',
                  top: '50%',
                  transform: 'translateY(-50%)',
                  background: 'none',
                  border: 'none',
                  color: showPassword ? '#00D2D3' : '#64748B',
                  cursor: 'pointer',
                  fontSize: '16px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  padding: '6px',
                  transition: 'color 0.2s ease',
                }}
              >
                {showPassword ? (
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24" />
                    <line x1="1" y1="1" x2="23" y2="23" />
                  </svg>
                ) : (
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
                    <circle cx="12" cy="12" r="3" />
                  </svg>
                )}
              </button>
            </div>
            {!isSignIn && (
              <span style={{ fontSize: '11px', color: '#64748B', marginTop: '6px', display: 'block' }}>
                Must be ≥8 chars with uppercase, digit & symbol (e.g. SkinSafe#2026)
              </span>
            )}
          </div>

          {!isSignIn && (
            <div>
              <label style={{ display: 'block', fontSize: '12px', fontWeight: 600, color: '#94A3B8', marginBottom: '6px' }}>
                Confirm Password
              </label>
              <div className="auth-input-container">
                <input
                  type={showConfirmPassword ? 'text' : 'password'}
                  name="confirm_security_key"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  placeholder="Confirm password"
                  required
                  autoComplete="new-password"
                  className="auth-input-field"
                />
                <button
                  type="button"
                  onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                  title={showConfirmPassword ? 'Hide password' : 'Show password'}
                  style={{
                    position: 'absolute',
                    right: '12px',
                    top: '50%',
                    transform: 'translateY(-50%)',
                    background: 'none',
                    border: 'none',
                    color: showConfirmPassword ? '#00D2D3' : '#64748B',
                    cursor: 'pointer',
                    fontSize: '16px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    padding: '6px',
                    transition: 'color 0.2s ease',
                  }}
                >
                  {showConfirmPassword ? (
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24" />
                      <line x1="1" y1="1" x2="23" y2="23" />
                    </svg>
                  ) : (
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
                      <circle cx="12" cy="12" r="3" />
                    </svg>
                  )}
                </button>
              </div>
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            style={{
              width: '100%',
              height: '48px',
              marginTop: '8px',
              borderRadius: '10px',
              border: 'none',
              background: 'linear-gradient(135deg, #00D2D3 0%, #0EA5E9 100%)',
              color: '#060A12',
              fontWeight: 800,
              fontSize: '14px',
              cursor: loading ? 'not-allowed' : 'pointer',
              boxShadow: '0 4px 20px rgba(0, 210, 211, 0.35)',
              transition: 'all 0.2s',
            }}
          >
            {loading ? 'Authenticating Access...' : (isSignIn ? 'Authenticate Clinician Access' : 'Register Clinical Account')}
          </button>
        </form>
      </div>
    </div>
  );
};

