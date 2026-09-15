import axios, { type InternalAxiosRequestConfig } from 'axios';
import type { PredictionResult, AuthResponse, DiagnosticHistoryItem } from '../types';

export const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://127.0.0.1:5000';
const API_BASE_URL = `${BACKEND_URL}/api/v1`;

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

apiClient.interceptors.request.use((config: InternalAxiosRequestConfig) => {
  const token = localStorage.getItem('access_token');
  if (token && config.headers) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export const predictImage = async (file: File): Promise<PredictionResult> => {
  const formData = new FormData();
  formData.append('image', file);

  const response = await apiClient.post<PredictionResult>('/predict', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

export const checkHealth = async () => {
  const response = await axios.get(`${BACKEND_URL}/api/health`, { timeout: 3000 });
  return response.data;
};

export const signIn = async (email: string, password: string): Promise<AuthResponse> => {
  const response = await axios.post<AuthResponse>(`${API_BASE_URL}/auth/signin`, { email, password }, { timeout: 8000 });
  if (response.data.access_token) {
    localStorage.setItem('access_token', response.data.access_token);
    localStorage.setItem('user_email', response.data.email);
    localStorage.setItem('user_name', response.data.name || 'Dr. Clinician');
    localStorage.setItem('user_role', response.data.role || 'Chief Dermatologist');
  }
  return response.data;
};

export const signUp = async (email: string, password: string, confirmPassword: string, name: string): Promise<AuthResponse> => {
  const response = await axios.post<AuthResponse>(`${API_BASE_URL}/auth/signup`, {
    email,
    password,
    confirm_password: confirmPassword,
    name
  }, { timeout: 8000 });
  if (response.data.access_token) {
    localStorage.setItem('access_token', response.data.access_token);
    localStorage.setItem('user_email', response.data.email);
    localStorage.setItem('user_name', response.data.name || name);
    localStorage.setItem('user_role', response.data.role || 'Dermatology Researcher');
  }
  return response.data;
};

export const logout = () => {
  localStorage.removeItem('access_token');
  localStorage.removeItem('user_email');
  localStorage.removeItem('user_name');
  localStorage.removeItem('user_role');
};

export const getScanHistory = async (search = '', status = 'all'): Promise<{ total_count: number; scans: DiagnosticHistoryItem[]; user_email: string }> => {
  const response = await apiClient.get<{ total_count: number; scans: DiagnosticHistoryItem[]; user_email: string }>('/history/scans', {
    params: { search, status },
    timeout: 5000
  });
  return response.data;
};

export const recordScanHistory = async (record: Partial<DiagnosticHistoryItem>): Promise<DiagnosticHistoryItem> => {
  const response = await apiClient.post<{ status: string; record: DiagnosticHistoryItem }>('/history/scans', record, { timeout: 5000 });
  return response.data.record;
};

export const deleteScanRecord = async (scanId: string): Promise<void> => {
  await apiClient.delete(`/history/scans/${scanId}`);
};
