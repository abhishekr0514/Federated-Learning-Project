import { apiGetPlatformCapabilities } from './api';

let inflightPromise = null;
let cachedReport = null;
let cachedAt = 0;
const TTL_MS = 30_000;

/**
 * Single-flight + short TTL cache for /api/v1/capabilities.
 */
export async function fetchPlatformCapabilities({ force = false } = {}) {
  const now = Date.now();
  if (!force && cachedReport && now - cachedAt < TTL_MS) {
    return cachedReport;
  }
  if (inflightPromise && !force) {
    return inflightPromise;
  }
  inflightPromise = apiGetPlatformCapabilities()
    .then((report) => {
      cachedReport = report;
      cachedAt = Date.now();
      inflightPromise = null;
      return report;
    })
    .catch((err) => {
      inflightPromise = null;
      throw err;
    });
  return inflightPromise;
}

export function invalidatePlatformCapabilitiesCache() {
  cachedReport = null;
  cachedAt = 0;
}
