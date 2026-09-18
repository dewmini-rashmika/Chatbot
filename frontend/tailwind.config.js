/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      typography: {
        invert: {
          css: {
            '--tw-prose-body': '#e5e7eb',
            '--tw-prose-headings': '#f9fafb',
            '--tw-prose-code': '#a78bfa',
          },
        },
      },
    },
  },
  plugins: [require('@tailwindcss/typography')],
}
