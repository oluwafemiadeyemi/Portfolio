import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Oluwafemi Adeyemi — Applied AI & Data Science Portfolio',
  description: '10 production-grade AI systems across healthcare, finance, retail, and HR. MIT Applied AI & Data Science.',
  keywords: ['machine learning', 'data science', 'AI portfolio', 'MIT', 'Parkinson detection', 'fraud detection'],
  authors: [{ name: 'Oluwafemi Adeyemi' }],
  openGraph: {
    title: 'Oluwafemi Adeyemi — Applied AI Portfolio',
    description: '10 enterprise AI systems. Live demos. Full source code.',
    type: 'website',
  },
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet" />
      </head>
      <body>{children}</body>
    </html>
  )
}
