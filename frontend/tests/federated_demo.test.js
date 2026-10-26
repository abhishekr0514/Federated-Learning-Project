import assert from 'node:assert/strict';
import {
  unwrapApiData,
  isFlowerRunActive,
  demoStartBlockedReason,
  connectionStatusBadge,
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

assert.equal(demoStartBlockedReason({ orchestration_enabled: true, training_enabled: true, security_configured: true, lifecycle: 'SERVER_STOPPED' }), null);

console.log('federated_demo.test.js: ok');
