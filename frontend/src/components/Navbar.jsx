import React from 'react';
import { Shield, LogOut, User, Menu, X, Building2 } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { HospitalContextSelector } from './HospitalContextSelector';

export function Navbar({ onToggleSidebar, isSidebarOpen }) {
  const { user, roles, logout } = useAuth();

  const primaryRole = user?.is_super_admin
    ? 'SUPER_ADMIN'
    : roles[0] || 'AUTHENTICATED_USER';

  return (
    <header className="navbar">
      <div className="navbar-left">
        <button
          className="btn-icon mobile-only"
          onClick={onToggleSidebar}
          aria-label="Toggle navigation menu"
        >
          {isSidebarOpen ? <X size={20} /> : <Menu size={20} />}
        </button>

        <div className="brand">
          <Shield className="brand-icon" size={24} />
          <span className="brand-title">MedShield-FL</span>
          <span className="brand-badge">v0.1.0</span>
        </div>
      </div>

      <div className="navbar-right">
        {roles.includes('HOSPITAL_ADMIN') && <HospitalContextSelector />}

        <div className="user-profile-badge">
          <div className="user-avatar">
            <User size={16} />
          </div>
          <div className="user-info">
            <span className="user-email">{user?.email || 'user@medshield.org'}</span>
            <span className="user-role-tag">{primaryRole}</span>
          </div>
        </div>

        <button className="btn btn-secondary btn-sm" onClick={logout} title="Sign out of session">
          <LogOut size={16} />
          <span className="desktop-only">Logout</span>
        </button>
      </div>
    </header>
  );
}
