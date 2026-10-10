import React from 'react';
import { AuthProvider, useAuth } from './context/AuthContext';
import { LoginPage } from './pages/auth/LoginPage';
import { SuperAdminDashboard } from './pages/admin/SuperAdminDashboard';
import { HospitalAdminDashboard } from './pages/hospital/HospitalAdminDashboard';
import { ClinicianDashboard } from './pages/clinician/ClinicianDashboard';
import { PatientDashboard } from './pages/patient/PatientDashboard';
import { LoadingSpinner, Forbidden403 } from './components/ErrorStates';

function MainAppRouter() {
  const { user, roles, isAuthenticated, loading } = useAuth();

  if (loading) {
    return <LoadingSpinner message="Authenticating MedShield-FL session..." />;
  }

  if (!isAuthenticated) {
    return <LoginPage />;
  }

  // Determine dashboard view based on primary canonical role
  if (user?.is_super_admin || roles.includes('SUPER_ADMIN')) {
    return <SuperAdminDashboard />;
  }

  if (roles.includes('HOSPITAL_ADMIN')) {
    return <HospitalAdminDashboard />;
  }

  if (roles.includes('CLINICIAN')) {
    return <ClinicianDashboard />;
  }

  if (roles.includes('PATIENT')) {
    return <PatientDashboard />;
  }

  // Fallback for authenticated user without an assigned role
  return (
    <div style={{ padding: '2rem' }}>
      <Forbidden403 requiredRole="Valid System Role (SUPER_ADMIN, HOSPITAL_ADMIN, CLINICIAN, or PATIENT)" />
    </div>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <MainAppRouter />
    </AuthProvider>
  );
}
