export type DiagnosticStatus = 'in_distribution' | 'ood';

export interface PredictionResult {
  status: DiagnosticStatus;
  predicted_class: string;
  full_class_name: string;
  confidence: number;
  diagnostic_certainty?: number;
  calibrated_confidence?: number;
  separation_margin?: number;
  entropy?: number;
  triage_label?: string;
  malignant_risk?: number;
  benign_risk?: number;
  probabilities: Record<string, number>;
  energy_score: number;
  image_url: string;
  gradcam_url: string;
}

export interface User {
  email: string;
  role: string;
  name?: string;
}

export interface AuthResponse {
  message: string;
  access_token: string;
  email: string;
  role?: string;
  name?: string;
}

export interface DiagnosticHistoryItem {
  id: string;
  timestamp: string;
  patientId: string;
  patientAge?: string;
  patientSex?: string;
  anatomSite?: string;
  predictedClass: string;
  confidence: number;
  status: DiagnosticStatus;
  imageUrl: string;
  gradcamUrl: string;
  energyScore?: number;
}

