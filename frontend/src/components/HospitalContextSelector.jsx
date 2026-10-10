import React from 'react';
import { Building2, ChevronDown } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

export function HospitalContextSelector() {
  const { hospitalMemberships, activeHospital, selectHospital } = useAuth();

  if (!hospitalMemberships || hospitalMemberships.length === 0) {
    return (
      <div className="hospital-badge">
        <Building2 size={16} />
        <span>Hospital Node: Unassigned</span>
      </div>
    );
  }

  if (hospitalMemberships.length === 1) {
    const hosp = hospitalMemberships[0].hospital || { name: 'Institutional Node', code: 'HOSP-01' };
    return (
      <div className="hospital-badge active">
        <Building2 size={16} />
        <span>{hosp.name || hosp.code}</span>
      </div>
    );
  }

  return (
    <div className="hospital-selector-wrapper">
      <div className="hospital-badge select">
        <Building2 size={16} />
        <select
          value={activeHospital?.id || ''}
          onChange={(e) => {
            const selected = hospitalMemberships.find((m) => m.hospital_id === e.target.value);
            if (selected) selectHospital(selected.hospital);
          }}
          className="hospital-dropdown"
        >
          {hospitalMemberships.map((mem) => (
            <option key={mem.id || mem.hospital_id} value={mem.hospital_id}>
              {mem.hospital?.name || `Hospital ${mem.hospital_id.substring(0, 8)}`}
            </option>
          ))}
        </select>
        <ChevronDown size={14} />
      </div>
    </div>
  );
}
