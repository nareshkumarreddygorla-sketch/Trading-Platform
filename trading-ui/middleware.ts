import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

/**
 * Middleware: protect dashboard routes at the edge.
 * Checks for auth indicator cookie (set by client on login).
 * Actual JWT validation happens on the API backend.
 */
const PUBLIC_PATHS = ["/login", "/register", "/_next", "/favicon.ico", "/api"];

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;

  // Allow public paths
  if (PUBLIC_PATHS.some((p) => pathname.startsWith(p))) {
    return NextResponse.next();
  }

  // Check for auth indicator cookie (set client-side on login)
  const hasAuth = request.cookies.get("has_auth")?.value === "1";
  if (!hasAuth && pathname !== "/login") {
    const loginUrl = new URL("/login", request.url);
    loginUrl.searchParams.set("redirect", pathname);
    return NextResponse.redirect(loginUrl);
  }

  // Add security headers
  const response = NextResponse.next();
  response.headers.set("X-Content-Type-Options", "nosniff");
  response.headers.set("X-Frame-Options", "DENY");
  response.headers.set("Referrer-Policy", "strict-origin-when-cross-origin");
  return response;
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
