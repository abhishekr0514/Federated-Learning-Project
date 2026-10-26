/**
 * SuperLink synthetic demo helpers (non-clinical). Parses /api/v1/federated/status envelopes.
 */

export function unwrapApiData(body) {
  if (body && typeof body === 'object' && 'data' in body && body.data != null) {
    return body.data;
  }
  return body ?? null;
}

export function isFlowerRunActive(lifecycle) {
  return ['STARTING', 'WAITING_FOR_CLIENTS', 'RUNNING', 'STOPPING'].includes(lifecycle);
}

export function flowerLifecycleLabel(lifecycle) {
  if (!lifecycle) return 'UNKNOWN';
  return String(lifecycle).replace(/_/g, ' ');
}

export function connectionStatusBadge(status) {
  const s = String(status || 'unknown').toLowerCase();
  if (s === 'online') return { label: 'Online', tone: 'success' };
  if (s === 'registered') return { label: 'Registered', tone: 'warning' };
  if (s === 'connecting') return { label: 'Connecting', tone: 'warning' };
  if (s === 'offline') return { label: 'Offline', tone: 'muted' };
  return { label: s, tone: 'muted' };
}

/** Live fleet connection vs participation in a completed demo run (process may be stopped). */
export function demoNodeStatusBadge(node) {
  if (node?.process_alive) {
    return connectionStatusBadge(node.connection_status);
  }
  if (node?.participation_status === 'completed_run') {
    return { label: 'Participated (stopped)', tone: 'success' };
  }
  return connectionStatusBadge('offline');
}

export function demoNodeProcessDetail(node) {
  if (node?.process_alive) return 'process running';
  if (node?.participation_status === 'completed_run') {
    return 'demo process stopped after successful run';
  }
  return 'process not running';
}

export function demoNodesSummaryLabel(orchestration) {
  if (!orchestration) return '—';
  const live = orchestration.demo_nodes_live_count ?? 0;
  const participated = orchestration.demo_nodes_participated_count ?? 0;
  const configured = orchestration.demo_supernodes?.length ?? 2;
  if (orchestration.post_run_processes_stopped) {
    return `${participated} participated in completed run · processes stopped · ${configured} configured`;
  }
  return `${live} live · ${configured} configured`;
}

export function demoStartBlockedReason(orchestration) {
  if (!orchestration) return 'Orchestration status not loaded yet.';
  if (!orchestration.orchestration_enabled) {
    return orchestration.demo_local_setup_hint || orchestration.reason;
  }
  if (!orchestration.training_enabled) {
    return orchestration.demo_local_setup_hint || orchestration.reason;
  }
  if (orchestration.tls_required && !orchestration.security_configured) {
    return (orchestration.security_blockers || []).join('; ') || orchestration.reason;
  }
  if (isFlowerRunActive(orchestration.lifecycle)) {
    return 'A SuperLink run is already in progress.';
  }
  return null;
}
