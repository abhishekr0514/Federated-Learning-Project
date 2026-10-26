import assert from 'node:assert/strict';
import {
  unwrapApiData,
  isFlowerRunActive,
  demoStartBlockedReason,
  connectionStatusBadge,
  demoNodeStatusBadge,
  demoNodesSummaryLabel,
  demoNodeProcessDetail,
} from '../src/utils/federatedDemo.js';

assert.deepEqual(unwrapApiData({ data: { lifecycle: 'RUNNING' } }), { lifecycle: 'RUNNING' });
assert.equal(isFlowerRunActive('STARTING'), true);
assert.equal(isFlowerRunActive('COMPLETED'), false);

const blocked = demoStartBlockedReason({
  orchestration_enabled: false,
  training_enabled: false,
  reason: 'off',
  demo_local_setup_hint: 'enable flags',
});
assert.match(blocked, /enable flags/);

assert.equal(connectionStatusBadge('online').label, 'Online');
assert.equal(connectionStatusBadge('offline').label, 'Offline');

assert.equal(
  demoNodeStatusBadge({ process_alive: false, participation_status: 'completed_run' }).label,
  'Participated (stopped)',
);
assert.match(
  demoNodeProcessDetail({ process_alive: false, participation_status: 'completed_run' }),
  /stopped after successful run/,
);

assert.match(
  demoNodesSummaryLabel({
    post_run_processes_stopped: true,
    demo_nodes_live_count: 0,
    demo_nodes_participated_count: 2,
    demo_supernodes: [{}, {}],
  }),
  /participated in completed run/,
);

assert.equal(demoStartBlockedReason({ orchestration_enabled: true, training_enabled: true, security_configured: true, lifecycle: 'SERVER_STOPPED' }), null);

console.log('federated_demo.test.js: ok');
