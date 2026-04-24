const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

function extractErrorMessage(payload, statusText) {
  if (!payload) return statusText || 'Request failed.';
  if (typeof payload === 'string') return payload;
  if (payload.detail) {
    if (typeof payload.detail === 'string') return payload.detail;
    if (typeof payload.detail === 'object' && payload.detail.message) return payload.detail.message;
  }
  if (payload.message) return payload.message;
  return statusText || 'Request failed.';
}

async function parseResponse(response) {
  const contentType = response.headers.get('content-type') || '';
  const payload = contentType.includes('application/json')
    ? await response.json()
    : await response.text();

  if (!response.ok) {
    throw new Error(extractErrorMessage(payload, response.statusText));
  }
  return payload;
}

function withAuthHeaders(token, includeContentType = false) {
  const headers = { Authorization: `Bearer ${token}` };
  if (includeContentType) headers['Content-Type'] = 'application/json';
  return headers;
}

export async function listDatabases(token) {
  const response = await fetch(`${API_URL}/databases`, {
    headers: withAuthHeaders(token),
  });
  return parseResponse(response);
}

export async function registerDatabase(token, payload) {
  const response = await fetch(`${API_URL}/databases/register`, {
    method: 'POST',
    headers: withAuthHeaders(token, true),
    body: JSON.stringify(payload),
  });
  return parseResponse(response);
}

export async function deleteDatabase(token, databaseId) {
  const response = await fetch(`${API_URL}/databases/${databaseId}`, {
    method: 'DELETE',
    headers: withAuthHeaders(token),
  });
  return parseResponse(response);
}

export async function updateDatabaseNickname(token, databaseId, nickname) {
  const response = await fetch(`${API_URL}/databases/${databaseId}`, {
    method: 'PATCH',
    headers: withAuthHeaders(token, true),
    body: JSON.stringify({ nickname }),
  });
  return parseResponse(response);
}

export async function resetAll(token) {
  const response = await fetch(`${API_URL}/reset`, {
    method: 'POST',
    headers: withAuthHeaders(token),
  });
  return parseResponse(response);
}
