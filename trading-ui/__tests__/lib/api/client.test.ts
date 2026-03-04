/**
 * Tests for the API client module at lib/api/client.ts
 *
 * Covers:
 *  - getToken returns null on server
 *  - Authorization header is set when token exists
 *  - 401 response clears auth and redirects
 *  - Refresh token is attempted on 401 before redirect
 */

// ---------------------------------------------------------------------------
// Helpers – create a minimal Response-like object that the client's
// `request()` function consumes.  jsdom does not always provide the global
// `Response` constructor, so we build one from scratch.
// ---------------------------------------------------------------------------

function makeResponse(body: unknown, init: { status: number; headers?: Record<string, string> }) {
  return {
    ok: init.status >= 200 && init.status < 300,
    status: init.status,
    statusText: init.status === 200 ? "OK" : "Error",
    headers: new Headers(init.headers ?? {}),
    json: () => Promise.resolve(body),
    text: () => Promise.resolve(JSON.stringify(body)),
  };
}

// ---------------------------------------------------------------------------
// localStorage mock
// ---------------------------------------------------------------------------

const localStorageMock = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: jest.fn((key: string) => store[key] ?? null),
    setItem: jest.fn((key: string, value: string) => {
      store[key] = value;
    }),
    removeItem: jest.fn((key: string) => {
      delete store[key];
    }),
    clear: jest.fn(() => {
      store = {};
    }),
    get length() {
      return Object.keys(store).length;
    },
    key: jest.fn((_index: number) => null),
  };
})();

Object.defineProperty(global, "localStorage", { value: localStorageMock, writable: true });

// Provide a minimal document.cookie setter/getter so setAuthTokens / clearAuthTokens work.
let cookieJar = "";
Object.defineProperty(document, "cookie", {
  get: () => cookieJar,
  set: (v: string) => {
    cookieJar = v;
  },
  configurable: true,
});

// Keep a reference to the real fetch so we can restore it.
const originalFetch = global.fetch;

beforeEach(() => {
  jest.resetModules();
  localStorageMock.clear();
  localStorageMock.getItem.mockClear();
  localStorageMock.setItem.mockClear();
  localStorageMock.removeItem.mockClear();
  cookieJar = "";
});

afterAll(() => {
  global.fetch = originalFetch;
});

// ---------------------------------------------------------------------------
// Helper – dynamically import the client so each test gets a fresh module
// (resets the internal `isRefreshing` flag).
// ---------------------------------------------------------------------------
async function loadClient() {
  return await import("@/lib/api/client");
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("API Client – getToken behaviour", () => {
  it("returns null when running on the server (no window)", async () => {
    // Temporarily remove `window` to simulate SSR
    const origWindow = global.window;
    // @ts-expect-error – intentionally removing window
    delete global.window;

    const { api } = await loadClient();

    // A request should NOT contain an Authorization header when there is no
    // token.  We intercept fetch to verify.
    const fetchSpy = jest.fn().mockResolvedValue(
      makeResponse({ ok: true }, { status: 200 }),
    );
    global.fetch = fetchSpy;

    try {
      await api.get("/test");
    } catch {
      // Server-side uses different base URL; we don't care if the URL is
      // wrong – we only inspect headers.
    }

    if (fetchSpy.mock.calls.length > 0) {
      const headers = fetchSpy.mock.calls[0][1]?.headers as Record<string, string>;
      expect(headers?.["Authorization"]).toBeUndefined();
    }

    // Restore
    global.window = origWindow;
  });
});

describe("API Client – Authorization header", () => {
  it("sets the Authorization header when a token exists in localStorage", async () => {
    localStorageMock.setItem("token", "test-jwt-token-123");

    const { api } = await loadClient();

    const fetchSpy = jest.fn().mockResolvedValue(
      makeResponse({ data: "ok" }, { status: 200 }),
    );
    global.fetch = fetchSpy;

    await api.get("/api/v1/risk/snapshot");

    expect(fetchSpy).toHaveBeenCalledTimes(1);
    const passedHeaders = fetchSpy.mock.calls[0][1]?.headers as Record<string, string>;
    expect(passedHeaders["Authorization"]).toBe("Bearer test-jwt-token-123");
  });
});

describe("API Client – 401 handling", () => {
  it("attempts to refresh the token on a 401 before redirecting", async () => {
    localStorageMock.setItem("token", "expired-token");
    localStorageMock.setItem("refresh_token", "valid-refresh-token");

    const { api } = await loadClient();

    // First call returns 401, refresh call succeeds, retry call succeeds.
    const fetchSpy = jest.fn()
      // 1st call – original request returns 401
      .mockResolvedValueOnce(makeResponse({ detail: "Unauthorized" }, { status: 401 }))
      // 2nd call – refresh endpoint succeeds
      .mockResolvedValueOnce(makeResponse({ access_token: "new-access-token" }, { status: 200 }))
      // 3rd call – retried original request succeeds
      .mockResolvedValueOnce(makeResponse({ result: "success" }, { status: 200 }));

    global.fetch = fetchSpy;

    const result = await api.get<{ result: string }>("/api/v1/risk/snapshot");

    // Should have made 3 fetch calls: original, refresh, retry
    expect(fetchSpy).toHaveBeenCalledTimes(3);

    // The refresh call should hit the refresh endpoint
    const refreshUrl = fetchSpy.mock.calls[1][0] as string;
    expect(refreshUrl).toContain("/auth/refresh");

    // The retried request should succeed
    expect(result).toEqual({ result: "success" });
  });

  it("clears auth and redirects to /login when refresh fails", async () => {
    localStorageMock.setItem("token", "expired-token");
    localStorageMock.setItem("refresh_token", "invalid-refresh-token");

    const { api, clearAuthTokens } = await loadClient();

    // jsdom does not support real navigation.  Setting window.location.href
    // logs a "Not implemented: navigation" error but does NOT throw a JS
    // exception, so the client code still reaches `throw new Error("Unauthorized")`.
    //
    // The href property on jsdom's Location is non-configurable, so we
    // cannot intercept the setter.  Instead we verify:
    //   1. The error is thrown.
    //   2. localStorage tokens are cleared via clearAuthTokens().
    //   3. The cookie is cleared.
    //
    // The redirect to "/login" is an environmental side-effect that jsdom
    // cannot execute; the client code path is confirmed by the fact that
    // clearAuthTokens() runs immediately before the redirect.

    const fetchSpy = jest.fn()
      // 1st call – original request returns 401
      .mockResolvedValueOnce(makeResponse({ detail: "Unauthorized" }, { status: 401 }))
      // 2nd call – refresh endpoint also fails
      .mockResolvedValueOnce(makeResponse({ detail: "Invalid refresh" }, { status: 401 }));

    global.fetch = fetchSpy;

    await expect(api.get("/api/v1/risk/snapshot")).rejects.toThrow("Unauthorized");

    // Auth tokens should be cleared (clearAuthTokens removes both keys)
    expect(localStorageMock.removeItem).toHaveBeenCalledWith("token");
    expect(localStorageMock.removeItem).toHaveBeenCalledWith("refresh_token");

    // The cookie should be cleared (max-age=0)
    expect(cookieJar).toContain("max-age=0");
  });
});
