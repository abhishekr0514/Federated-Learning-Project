import React, { useEffect, useState } from 'react';
import {
  LayoutDashboard,
  Building2,
  Users,
  UserCheck,
  FileText,
  Activity,
  Cpu,
  BarChart3,
  ShieldCheck,
  History,
  HelpCircle,
  Stethoscope,
  Heart,
  Radio,
  FileSpreadsheet,
  Bell,
  MessageSquare,
} from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { capabilityNavBadge } from '../utils/capabilityNav';
import { fetchPlatformCapabilities } from '../services/capabilityClient';

export function Sidebar({ isOpen, activeTab, onSelectTab, role, patientPortalBadges = null }) {
  const { user, permissions } = useAuth();
  const [capabilityBadges, setCapabilityBadges] = useState(null);

  const refreshCapabilityBadges = React.useCallback(async () => {
    try {
      const report = await fetchPlatformCapabilities({ force: true });
      const caps = report?.capabilities || report;
      setCapabilityBadges({
        gat: capabilityNavBadge(caps?.multimodal_gat_fusion),
        dice: capabilityNavBadge(caps?.dice_counterfactuals),
        dowhy: capabilityNavBadge(caps?.dowhy_causal),
        flower: capabilityNavBadge(caps?.flower_federated),
      });
    } catch {
      setCapabilityBadges({
        gat: 'ERR',
        dice: 'ERR',
        dowhy: 'ERR',
        flower: 'ERR',
      });
    }
  }, []);

  useEffect(() => {
    const rolesWithCaps = ['CLINICIAN', 'SUPER_ADMIN', 'HOSPITAL_ADMIN'];
    if (!rolesWithCaps.includes(role)) {
      return undefined;
    }
    refreshCapabilityBadges();
    const onFederate = () => refreshCapabilityBadges();
    window.addEventListener('medshield:federate_changed', onFederate);
    return () => window.removeEventListener('medshield:federate_changed', onFederate);
  }, [role, refreshCapabilityBadges]);

  const renderNavGroup = (title, items) => (
    <div className="nav-group" key={title}>
      <div className="nav-group-title">{title}</div>
      {items.map((item) => {
        const Icon = item.icon;
        const isActive = activeTab === item.id;
        return (
          <button
            key={item.id}
            className={`nav-item ${isActive ? 'active' : ''}`}
            onClick={() => onSelectTab(item.id)}
          >
            <Icon size={18} className="nav-icon" />
            <span className="nav-label">{item.label}</span>
            {item.badge && <span className="badge badge-sm">{item.badge}</span>}
          </button>
        );
      })}
    </div>
  );

  let navConfig = [];

  if (role === 'SUPER_ADMIN' || user?.is_super_admin) {
    navConfig = [
      {
        group: 'Global Administration',
        items: [
          { id: 'overview', label: 'System Overview', icon: LayoutDashboard },
          { id: 'hospitals', label: 'Hospital Nodes', icon: Building2 },
          { id: 'federated', label: 'Federated Engine', icon: Cpu, badge: capabilityBadges?.flower || 'Flower' },
          { id: 'models', label: 'Model Artifacts', icon: Activity },
          { id: 'metrics', label: 'Global Metrics', icon: BarChart3 },
        ],
      },
      {
        group: 'Security & Compliance',
        items: [
          { id: 'security', label: 'Security Monitor', icon: ShieldCheck },
          { id: 'audit', label: 'Audit Log Ledger', icon: History },
        ],
      },
    ];
  } else if (role === 'HOSPITAL_ADMIN') {
    navConfig = [
      {
        group: 'Node Administration',
        items: [
          { id: 'overview', label: 'Node Overview', icon: LayoutDashboard },
          { id: 'system_health', label: 'System Health & DB', icon: Activity },
          { id: 'clinicians', label: 'Clinician Directory', icon: UserCheck },
          { id: 'patients', label: 'Opaque Patients', icon: Users },
        ],
      },
      {
        group: 'Models & Engine',
        items: [
          { id: 'inference', label: 'Diagnostic Studio', icon: Activity, badge: 'Live ML' },
          { id: 'models', label: 'Model Artifacts', icon: Cpu },
          { id: 'metrics', label: 'Local Metrics', icon: BarChart3 },
          { id: 'federated', label: 'Federated Node', icon: Radio, badge: capabilityBadges?.flower || '…' },
        ],
      },
      {
        group: 'Privacy & Governance',
        items: [
          { id: 'privacy', label: 'Privacy Telemetry', icon: ShieldCheck },
          { id: 'data_quality', label: 'Data Quality', icon: FileSpreadsheet },
          { id: 'audit', label: 'Node Audit Trail', icon: History },
          { id: 'alerts', label: 'Operational Alerts', icon: Bell },
        ],
      },
    ];
  } else if (role === 'CLINICIAN') {
    const cb = capabilityBadges;
    navConfig = [
      {
        group: 'Clinical Workspace',
        items: [
          { id: 'dashboard', label: 'Clinician Workspace', icon: LayoutDashboard },
          { id: 'inference', label: 'Diagnostic Studio', icon: Activity, badge: 'Live ML' },
          { id: 'patients', label: 'Assigned Patients', icon: Stethoscope },
          { id: 'documents', label: 'Clinical Reports', icon: FileText },
          { id: 'questions', label: 'Patient Questions', icon: MessageSquare },
        ],
      },
      {
        group: 'Diagnostic Intelligence',
        items: [
          { id: 'federated', label: 'Federated Node', icon: Radio, badge: cb?.flower || '…' },
          { id: 'diagnosis', label: 'GAT Diagnosis', icon: Activity, badge: cb?.gat || '…' },
          // { id: 'explanations', label: 'DiCE Explanations', icon: Cpu, badge: cb?.dice || '…' },
          { id: 'causal', label: 'DoWhy Causal AI', icon: BarChart3, badge: cb?.dowhy || '…' },
        ],
      },
    ];
  } else if (role === 'PATIENT') {
    const pb = patientPortalBadges || {};
    navConfig = [
      {
        group: 'Personal Health Portal',
        items: [
          { id: 'health', label: 'My Health', icon: Heart },
          { id: 'records', label: 'My Records', icon: FileText, badge: pb.records },
          { id: 'reports', label: 'My Reports', icon: Activity, badge: pb.reports },
          {
            id: 'assessment',
            label: 'My Assessment',
            icon: Stethoscope,
            badge: pb.assessment || 'Not enabled',
          },
          { id: 'results', label: 'My Results', icon: FileSpreadsheet, badge: pb.results },
          {
            id: 'questions',
            label: 'Questions for My Clinician',
            icon: HelpCircle,
            badge: pb.messaging || 'Not built',
          },
        ],
      },
    ];
  }

  return (
    <aside className={`sidebar ${isOpen ? 'open' : ''}`}>
      <div className="sidebar-nav">
        {navConfig.map((group) => renderNavGroup(group.group, group.items))}
      </div>

      <div className="sidebar-footer">
        <div className="privacy-notice-box">
          <ShieldCheck size={14} style={{ display: 'inline', marginRight: '4px' }} />
          <strong>Privacy Boundary Active</strong>
          <p>Zero raw patient PII transmitted across hospital boundaries.</p>
        </div>
      </div>
    </aside>
  );
}
