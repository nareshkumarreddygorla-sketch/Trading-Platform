import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "hsl(222 47% 8%)",
        foreground: "hsl(210 40% 98%)",
        card: "hsl(222 47% 11%)",
        "card-foreground": "hsl(210 40% 98%)",
        primary: "hsl(217 91% 60%)",
        "primary-foreground": "hsl(222 47% 11%)",
        profit: "hsl(142 76% 36%)",
        loss: "hsl(0 84% 60%)",
        warning: "hsl(38 92% 50%)",
        "safe-mode": "hsl(25 95% 53%)",
        muted: "hsl(217 33% 17%)",
        "muted-foreground": "hsl(215 20% 65%)",
        border: "hsl(217 33% 17%)",
      },
      fontFamily: {
        sans: ["var(--font-sans)", "system-ui", "sans-serif"],
        mono: ["ui-monospace", "monospace"],
      },
      animation: {
        "pulse-slow": "pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite",
      },
      boxShadow: {
        "glow-green": "0 0 20px rgba(34, 197, 94, 0.4)",
        "glow-red": "0 0 20px rgba(239, 68, 68, 0.4)",
      },
    },
  },
  plugins: [],
};

export default config;
