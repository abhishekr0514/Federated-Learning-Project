import React from 'react';
import { useAuth } from '../context/AuthContext';
import { LoadingSpinner, Forbidden403, Unauthorized401 } from './ErrorStates';

export function ProtectedRoute({ children, allowedRoles, requiredPermission }) {
  const { user, loading, isAuthenticated, hasRole, hasPermission } = useAuth();

  if (loading) {
    return <LoadingSpinner message="Verifying session authorization..." />;
  }

  if (!isAuthenticated) {
    return <Unauthorized401 />;
  }

  // Check role requirement
  if (allowedRoles && allowedRoles.length > 0) {
    const isAllowed = allowedRoles.some((role) => hasRole(role));
    if (!isAllowed) {
      return <Forbidden403 requiredRole={allowedRoles.join(' or ')} />;
    }
  }

  // Check permission requirement
  if (requiredPermission) {
    if (!hasPermission(requiredPermission)) {
      return <Forbidden403 requiredRole={`Permission: ${requiredPermission}`} />;
    }
  }

  return children;
}
