/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{js,jsx,ts,tsx}',
    '../src/**/*.{js,jsx,ts,tsx}', // allow classes from repo-level src
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
