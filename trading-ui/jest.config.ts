import type { Config } from "@jest/types";

const config: Config.InitialOptions = {
  testEnvironment: "jsdom",
  roots: ["<rootDir>/__tests__"],
  transform: {
    "^.+\\.tsx?$": [
      "ts-jest",
      {
        tsconfig: "tsconfig.json",
        // Use ESM-compatible module resolution for ts-jest
        useESM: false,
      },
    ],
  },
  moduleNameMapper: {
    // Map the @/* path alias used by Next.js to the project root
    "^@/(.*)$": "<rootDir>/$1",
  },
  moduleFileExtensions: ["ts", "tsx", "js", "jsx", "json"],
  testMatch: ["**/__tests__/**/*.test.ts", "**/__tests__/**/*.test.tsx"],
  // Ignore the .next build output and node_modules
  testPathIgnorePatterns: ["/node_modules/", "/.next/"],
};

export default config;
