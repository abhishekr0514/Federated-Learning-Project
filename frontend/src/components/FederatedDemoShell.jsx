import React from 'react';
import { useAuth } from '../context/AuthContext';
import { FederatedDemoAccessGate } from './FederatedDemoAccessGate';
import { FederatedLearningPanel } from './FederatedLearningPanel';

/** Renders SuperLink panel only when the session has federated:monitor (Super Admin). */
export function FederatedDemoShell() {
  const { hasPermission, loading } = useAuth();

  if (loading) {
    return null;
  }

  if (!hasPermission('federated.monitor')) {
    return <FederatedDemoAccessGate />;
  }

  return <FederatedLearningPanel />;
}
