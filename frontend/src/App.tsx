import { useState, useEffect } from 'react';
import { Dropzone } from './components/Dropzone';
import { ImageComparisonSlider } from './components/ImageComparisonSlider';
import { ProbabilityGauge } from './components/ProbabilityGauge';
import { OODAlertBanner } from './components/OODAlertBanner';
import { AuthModal } from './components/AuthModal';
import { ScanHistoryTab } from './components/ScanHistoryTab';
import { predictImage, logout, recordScanHistory, BACKEND_URL } from './services/api';
import type { PredictionResult, User } from './types';


export function App() {
  const [activeTab, setActiveTab] = useState<'scanner' | 'history'>('scanner');
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [isAuthModalOpen, setIsAuthModalOpen] = useState<boolean>(false);
  const [currentUser, setCurrentUser] = useState<User | null>(null);

  // Patient Clinical Metadata
  const [patientId, setPatientId] = useState<string>('PT-10492');
  const [patientAge, setPatientAge] = useState<string>('55');
  const [patientSex, setPatientSex] = useState<string>('male');
  const [anatomSite, setAnatomSite] = useState<string>('anterior torso');

  useEffect(() => {
    const savedEmail = localStorage.getItem('user_email');
    const savedName = localStorage.getItem('user_name');
    const savedRole = localStorage.getItem('user_role');
    if (savedEmail) {
      setCurrentUser({
        email: savedEmail,
        name: savedName || 'Dr. Clinician',
        role: savedRole || 'Chief Dermatologist',
      });
    }
  }, []);

  const handleImageSelected = async (file: File) => {
    if (!currentUser) {
      setIsAuthModalOpen(true);
      return;
    }

    setIsLoading(true);
    setError(null);
    try {
      const data = await predictImage(file);
      setResult(data);

      // Persist to Backend Scan History
      try {
        await recordScanHistory({
          patientId: patientId.trim() || undefined,
          patientAge,
          patientSex,
          anatomSite,
          predictedClass: data.full_class_name,
          confidence: data.confidence,
          status: data.status,
          energyScore: data.energy_score,
          imageUrl: data.image_url,
          gradcamUrl: data.gradcam_url,
        });
      } catch (saveErr) {
        console.warn('Could not sync scan to history backend:', saveErr);
      }
    } catch (err: any) {
      if (err.code === 'ERR_NETWORK' || !err.response) {
        setError('Inference service is currently offline. Please ensure run_server.py is running.');
      } else {
        setError(err?.response?.data?.error || err.message || 'Error processing dermoscopic image.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleLogout = () => {
    logout();
    setCurrentUser(null);
    setIsAuthModalOpen(false);
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', background: '#060A12' }}>
      {/* Clinician Authentication Modal */}
      <AuthModal
        isOpen={isAuthModalOpen}
        onClose={() => setIsAuthModalOpen(false)}
        onSuccess={(email, name, role) => {
          setCurrentUser({ email, name, role });
        }}
      />

      {/* Top Obsidian Medical Navbar */}
      <header
        style={{
          borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
          background: 'rgba(11, 17, 30, 0.95)',
          backdropFilter: 'blur(20px)',
          position: 'sticky',
          top: 0,
          zIndex: 50,
          padding: '12px 32px',
        }}
      >
        <div style={{ maxWidth: '1440px', margin: '0 auto', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
          {/* Brand & Version */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
            <div
              style={{
                width: '38px',
                height: '38px',
                borderRadius: '10px',
                background: 'linear-gradient(135deg, #00D2D3 0%, #0EA5E9 100%)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                boxShadow: '0 0 20px rgba(0, 210, 211, 0.35)',
              }}
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#060A12" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>
              </svg>
            </div>
            <div>
              <h1 style={{ fontSize: '17px', fontWeight: 800, color: '#FFFFFF', letterSpacing: '-0.02em', lineHeight: 1.2 }}>
                SkinSafe <span style={{ color: '#00D2D3', fontWeight: 500 }}>AI Workstation</span>
              </h1>
              <p style={{ fontSize: '12px', color: '#94A3B8' }}>Clinical Diagnostic Support & Research Intelligence</p>
            </div>
          </div>

          {/* Center Navigation Segment */}
          <nav style={{ display: 'flex', gap: '4px', background: 'rgba(6, 10, 18, 0.8)', padding: '4px', borderRadius: '10px', border: '1px solid rgba(255, 255, 255, 0.08)' }}>
            {[
              { id: 'scanner', label: '🔬 Diagnostic Scanner' },
              { id: 'history', label: '📋 Scan Registry & Audit' },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                style={{
                  padding: '7px 15px',
                  borderRadius: '8px',
                  border: 'none',
                  background: activeTab === tab.id ? 'linear-gradient(135deg, #00D2D3 0%, #0EA5E9 100%)' : 'transparent',
                  color: activeTab === tab.id ? '#060A12' : '#94A3B8',
                  fontWeight: 700,
                  fontSize: '13px',
                  cursor: 'pointer',
                  transition: 'all 0.2s',
                }}
              >
                {tab.label}
              </button>
            ))}
          </nav>

          {/* Right Status & Clinician Controls */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>

            {/* Clinician Profile */}
            {currentUser ? (
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <div style={{ textAlign: 'right' }}>
                  <div style={{ fontSize: '13px', fontWeight: 700, color: '#F8FAFC' }}>{currentUser.name}</div>
                  <div style={{ fontSize: '11px', color: '#00D2D3' }}>{currentUser.role}</div>
                </div>
                <button
                  onClick={handleLogout}
                  style={{
                    padding: '6px 12px',
                    borderRadius: '8px',
                    background: 'rgba(255, 255, 255, 0.06)',
                    border: '1px solid rgba(255, 255, 255, 0.12)',
                    color: '#94A3B8',
                    fontSize: '12px',
                    fontWeight: 600,
                    cursor: 'pointer',
                    transition: 'all 0.2s',
                  }}
                >
                  Sign Out
                </button>
              </div>
            ) : (
              <button
                onClick={() => setIsAuthModalOpen(true)}
                style={{
                  padding: '8px 16px',
                  borderRadius: '8px',
                  background: 'linear-gradient(135deg, #00D2D3 0%, #0EA5E9 100%)',
                  border: 'none',
                  color: '#060A12',
                  fontWeight: 800,
                  fontSize: '13px',
                  cursor: 'pointer',
                  boxShadow: '0 4px 16px rgba(0, 210, 211, 0.3)',
                  transition: 'all 0.2s',
                }}
              >
                🔐 Clinician Sign In
              </button>
            )}
          </div>
        </div>
      </header>


      {/* Main Workspace */}
      <main style={{ flex: 1, maxWidth: '1440px', width: '100%', margin: '0 auto', padding: '32px' }}>
        {error && (
          <div
            style={{
              marginBottom: '20px',
              padding: '12px 18px',
              background: 'rgba(244, 63, 94, 0.12)',
              border: '1px solid rgba(244, 63, 94, 0.35)',
              borderRadius: '10px',
              color: '#FDA4AF',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              fontSize: '13px',
            }}
          >
            <span>{error}</span>
            <button
              onClick={() => setError(null)}
              style={{ background: 'none', border: 'none', color: '#FDA4AF', cursor: 'pointer', fontSize: '16px' }}
            >
              ✕
            </button>
          </div>
        )}

        {/* Tab 1: Diagnostic Scanner */}
        {activeTab === 'scanner' && (
          <div style={{ display: 'grid', gridTemplateColumns: result ? '1fr 1fr' : '1fr', gap: '32px', alignItems: 'start' }}>
            {/* Left Column */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
              <div className="glass-panel" style={{ padding: '28px' }}>
                <h2 style={{ fontSize: '18px', fontWeight: 700, color: '#FFFFFF', marginBottom: '16px' }}>
                  Dermoscopy Image Ingestion
                </h2>
                <Dropzone
                  onImageSelected={handleImageSelected}
                  isLoading={isLoading}
                  isAuthenticated={!!currentUser}
                  onRequireAuth={() => setIsAuthModalOpen(true)}
                />
              </div>


              {/* Patient Metadata Form */}
              <div className="glass-panel" style={{ padding: '24px' }}>
                <h3 style={{ fontSize: '15px', fontWeight: 700, color: '#FFFFFF', marginBottom: '14px' }}>
                  Patient Clinical Context (Optional Metadata)
                </h3>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '14px' }}>
                  <div>
                    <label style={{ display: 'block', fontSize: '12px', fontWeight: 600, color: '#94A3B8', marginBottom: '6px' }}>Patient ID / MRN</label>
                    <input
                      type="text"
                      placeholder="e.g. PT-10492"
                      value={patientId}
                      onChange={(e) => setPatientId(e.target.value)}
                      style={{ width: '100%', padding: '9px 12px', background: 'rgba(8, 12, 20, 0.85)', border: '1px solid rgba(255, 255, 255, 0.1)', borderRadius: '8px', color: '#F8FAFC', fontSize: '13px' }}
                    />
                  </div>
                  <div>
                    <label style={{ display: 'block', fontSize: '12px', fontWeight: 600, color: '#94A3B8', marginBottom: '6px' }}>Approx. Age</label>
                    <input
                      type="number"
                      value={patientAge}
                      onChange={(e) => setPatientAge(e.target.value)}
                      style={{ width: '100%', padding: '9px 12px', background: 'rgba(8, 12, 20, 0.85)', border: '1px solid rgba(255, 255, 255, 0.1)', borderRadius: '8px', color: '#F8FAFC', fontSize: '13px' }}
                    />
                  </div>
                  <div>
                    <label style={{ display: 'block', fontSize: '12px', fontWeight: 600, color: '#94A3B8', marginBottom: '6px' }}>Biological Sex</label>
                    <select
                      value={patientSex}
                      onChange={(e) => setPatientSex(e.target.value)}
                      style={{ width: '100%', padding: '9px 12px', background: 'rgba(8, 12, 20, 0.85)', border: '1px solid rgba(255, 255, 255, 0.1)', borderRadius: '8px', color: '#F8FAFC', fontSize: '13px' }}
                    >
                      <option value="male">Male</option>
                      <option value="female">Female</option>
                      <option value="unknown">Unknown</option>
                    </select>
                  </div>
                  <div>
                    <label style={{ display: 'block', fontSize: '12px', fontWeight: 600, color: '#94A3B8', marginBottom: '6px' }}>Anatomical Site</label>
                    <select
                      value={anatomSite}
                      onChange={(e) => setAnatomSite(e.target.value)}
                      style={{ width: '100%', padding: '9px 12px', background: 'rgba(8, 12, 20, 0.85)', border: '1px solid rgba(255, 255, 255, 0.1)', borderRadius: '8px', color: '#F8FAFC', fontSize: '13px' }}
                    >
                      <option value="anterior torso">Anterior Torso</option>
                      <option value="posterior torso">Posterior Torso</option>
                      <option value="lower extremity">Lower Extremity</option>
                      <option value="upper extremity">Upper Extremity</option>
                      <option value="head/neck">Head / Neck</option>
                      <option value="palms/soles">Palms / Soles</option>
                    </select>
                  </div>
                </div>
              </div>

              {/* Grad-CAM Heatmap Slider */}
              {result && (
                <div className="glass-panel" style={{ padding: '24px' }}>
                  <h2 style={{ fontSize: '18px', fontWeight: 700, color: '#FFFFFF', marginBottom: '16px' }}>
                    Visual Explainability (Grad-CAM Saliency Map)
                  </h2>
                  <ImageComparisonSlider
                    originalUrl={`${BACKEND_URL}${result.image_url}`}
                    gradcamUrl={`${BACKEND_URL}${result.gradcam_url}`}
                  />
                </div>
              )}
            </div>

            {/* Right Column: Diagnostic Assessment */}
            {result && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
                <OODAlertBanner
                  status={result.status}
                  energyScore={result.energy_score}
                />

                {/* Primary Diagnostic Assessment Card */}
                <div className="glass-panel" style={{ padding: '30px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '16px' }}>
                    <div>
                      <span style={{ fontSize: '11px', fontWeight: 700, color: '#94A3B8', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                        Primary AI Pathological Assessment
                      </span>
                      <h2 style={{ fontSize: '28px', fontWeight: 800, color: '#FFFFFF', marginTop: '6px', letterSpacing: '-0.02em' }}>
                        {result.full_class_name}
                      </h2>
                      {result.triage_label && (
                        <div style={{ marginTop: '8px', display: 'inline-flex', alignItems: 'center', gap: '6px', padding: '4px 12px', borderRadius: '20px', fontSize: '12px', fontWeight: 600, backgroundColor: result.triage_label.includes('Malignant') ? 'rgba(239, 68, 68, 0.15)' : result.triage_label.includes('Pre-cancer') ? 'rgba(245, 158, 11, 0.15)' : 'rgba(16, 185, 129, 0.15)', color: result.triage_label.includes('Malignant') ? '#EF4444' : result.triage_label.includes('Pre-cancer') ? '#F59E0B' : '#10B981', border: `1px solid ${result.triage_label.includes('Malignant') ? 'rgba(239, 68, 68, 0.3)' : result.triage_label.includes('Pre-cancer') ? 'rgba(245, 158, 11, 0.3)' : 'rgba(16, 185, 129, 0.3)'}` }}>
                          <span>●</span> {result.triage_label}
                        </div>
                      )}
                    </div>

                    <div style={{ textAlign: 'right' }}>
                      <span style={{ fontSize: '11px', fontWeight: 700, color: '#94A3B8', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                        Diagnostic Confidence
                      </span>
                      <div style={{ fontSize: '32px', fontWeight: 800, color: '#00D2D3', fontFamily: 'monospace' }}>
                        {((result.confidence) * 100).toFixed(1)}%
                      </div>
                      {result.separation_margin !== undefined && (
                        <span style={{ fontSize: '11px', color: '#64748B', display: 'block', marginTop: '2px' }}>
                          Margin: +{(result.separation_margin * 100).toFixed(1)}%
                        </span>
                      )}
                    </div>
                  </div>
                </div>

                <ProbabilityGauge
                  probabilities={result.probabilities}
                  predictedClass={result.predicted_class}
                />
              </div>
            )}
          </div>
        )}

        {/* Tab 2: Scan Registry */}
        {activeTab === 'history' && (
          <ScanHistoryTab
            onSelectScan={(item) => {
              if (item.patientId) setPatientId(item.patientId);
              if (item.patientAge) setPatientAge(item.patientAge);
              if (item.patientSex) setPatientSex(item.patientSex);
              if (item.anatomSite) setAnatomSite(item.anatomSite);
              setResult({
                status: item.status,
                predicted_class: item.predictedClass,
                full_class_name: item.predictedClass,
                confidence: item.confidence,
                probabilities: { [item.predictedClass]: item.confidence },
                energy_score: item.energyScore || 0,
                image_url: item.imageUrl,
                gradcam_url: item.gradcamUrl,
              });
              setActiveTab('scanner');
            }}
          />
        )}

      </main>
    </div>
  );
}

export default App;
