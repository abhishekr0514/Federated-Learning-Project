import assert from 'node:assert';
import { normalizeIntegrationStatus, parseAuthTokenEnvelope } from '../src/utils/integrationStatus.js';

const mapped = normalizeIntegrationStatus('CONFIGURATION_REQUIRED');
assert.strictEqual(mapped.status, 'CONFIGURATION_REQUIRED');
assert.strictEqual(mapped.label, 'CONFIG REQUIRED');

const envelope = parseAuthTokenEnvelope({ data: { access_token: 'a', refresh_token: 'b' } });
assert.strictEqual(envelope.access_token, 'a');
assert.strictEqual(envelope.refresh_token, 'b');

console.log('integration_status.test.js passed');
