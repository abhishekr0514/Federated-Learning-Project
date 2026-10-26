import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { apiRequest, ApiError } from '../services/api';
import { invalidatePlatformCapabilitiesCache } from '../services/capabilityClient';
import { parseAuthTokenEnvelope } from '../utils/integrationStatus';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [roles, setRoles] = useState([]);
  const [permissions, setPermissions] = useState([]);
  const [hospitalMemberships, setHospitalMemberships] = useState([]);
  const [activeHospital, setActiveHospital] = useState(null);
  const [patientRecord, setPatientRecord] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  /**
   * Fetch user account metadata (/api/v1/auth/me)
   */
  const fetchCurrentUser = useCallback(async () => {
    const accessToken = sessionStorage.getItem('medshield_access_token');
    if (!accessToken) {
      setUser(null);
      setRoles([]);
      setPermissions([]);
      setHospitalMemberships([]);
      setActiveHospital(null);
      setPatientRecord(null);
      setLoading(false);
      return;
    }

    try {
      setLoading(true);
      setError(null);
      const res = await apiRequest('/api/v1/auth/me');
      const userData = res.data;

      setUser(userData);
      const userRoles = userData.roles || [];
      setRoles(userRoles);

      // Infer frontend navigation permissions based on canonical roles
      const inferredPerms = new Set();
      if (userData.is_super_admin || userRoles.includes('SUPER_ADMIN')) {
        inferredPerms.add('global.admin');
        inferredPerms.add('hospitals.manage');
        inferredPerms.add('federated.monitor');
        inferredPerms.add('models.manage');
        inferredPerms.add('metrics.read');
        inferredPerms.add('security.monitor');
        inferredPerms.add('audit.read');
      }

      if (userRoles.includes('HOSPITAL_ADMIN')) {
        inferredPerms.add('hospital.config');
        inferredPerms.add('clinician.manage');
        inferredPerms.add('patient.local_manage');
        inferredPerms.add('local_training.monitor');
        inferredPerms.add('local_metrics.read');
        inferredPerms.add('local_audit.read');
      }

      if (userRoles.includes('CLINICIAN')) {
        inferredPerms.add('clinical.read');
        inferredPerms.add('clinical.write');
        inferredPerms.add('diagnosis.read');
        inferredPerms.add('explanations.read');
        inferredPerms.add('causal.read');
      }

      if (userRoles.includes('PATIENT')) {
        inferredPerms.add('patient.own_read');
        inferredPerms.add('diagnosis.own_read');
      }

      setPermissions(Array.from(inferredPerms));
    } catch (err) {
      console.error('Failed to resolve current user context:', err);
      setUser(null);
      setRoles([]);
      setPermissions([]);
      sessionStorage.removeItem('medshield_access_token');
      sessionStorage.removeItem('medshield_refresh_token');
    } finally {
      setLoading(false);
    }
  }, []);

  /**
   * Login credentials authentication
   */
  const login = async (email, password) => {
    setLoading(true);
    setError(null);
    sessionStorage.removeItem('medshield_access_token');
    sessionStorage.removeItem('medshield_refresh_token');
    setUser(null);
    setRoles([]);
    setPermissions([]);
    try {
      const res = await apiRequest('/api/v1/auth/login', {
        method: 'POST',
        body: JSON.stringify({ email, password }),
      });

      const { access_token, refresh_token } = parseAuthTokenEnvelope(res);
      sessionStorage.setItem('medshield_access_token', access_token);
      sessionStorage.setItem('medshield_refresh_token', refresh_token);

      invalidatePlatformCapabilitiesCache();
      await fetchCurrentUser();
      return res;
    } catch (err) {
      setError(err.message || 'Authentication failed');
      setLoading(false);
      throw err;
    }
  };

  /**
   * Session Logout
   */
  const logout = async () => {
    const refreshToken = sessionStorage.getItem('medshield_refresh_token');
    if (refreshToken) {
      try {
        await apiRequest('/api/v1/auth/logout', {
          method: 'POST',
          body: JSON.stringify({ refresh_token: refreshToken }),
        });
      } catch (err) {
        console.warn('Backend logout call warning:', err.message);
      }
    }

    sessionStorage.removeItem('medshield_access_token');
    sessionStorage.removeItem('medshield_refresh_token');
    invalidatePlatformCapabilitiesCache();
    setUser(null);
    setRoles([]);
    setPermissions([]);
    setHospitalMemberships([]);
    setActiveHospital(null);
    setPatientRecord(null);
    setLoading(false);
  };

  /**
   * Change active hospital context
   */
  const selectHospital = (hospital) => {
    setActiveHospital(hospital);
  };

  /**
   * Listen for global auth failure events (e.g. from failed refresh)
   */
  useEffect(() => {
    const handleAuthFailed = () => {
      logout();
    };
    window.addEventListener('medshield:auth_failed', handleAuthFailed);
    return () => window.removeEventListener('medshield:auth_failed', handleAuthFailed);
  }, []);

  useEffect(() => {
    fetchCurrentUser();
  }, [fetchCurrentUser]);

  const value = {
    user,
    roles,
    permissions,
    hospitalMemberships,
    activeHospital,
    patientRecord,
    loading,
    error,
    isAuthenticated: !!user,
    login,
    logout,
    selectHospital,
    refetchUser: fetchCurrentUser,
    hasRole: (role) => roles.includes(role) || (user?.is_super_admin && role === 'SUPER_ADMIN'),
    hasPermission: (perm) => permissions.includes(perm) || user?.is_super_admin,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
