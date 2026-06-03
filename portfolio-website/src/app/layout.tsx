import type { Metadata } from 'next'
import './globals.css'
import { Sidebar } from '@/components/sidebar'

export const metadata: Metadata = {
  title: 'Oluwafemi Adeyemi — Applied AI & Data Science Portfolio',
  description: '17 production-grade AI systems across healthcare, finance, retail, HR, and entertainment. MIT Applied AI & Data Science.',
  keywords: ['machine learning', 'data science', 'AI portfolio', 'MIT', 'Parkinson detection', 'fraud detection'],
  authors: [{ name: 'Oluwafemi Adeyemi' }],
  openGraph: {
    title: 'Oluwafemi Adeyemi — Applied AI Portfolio',
    description: '17 enterprise AI systems. Live demos. Real datasets. Full source code.',
    type: 'website',
  },
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,300;0,14..32,400;0,14..32,500;0,14..32,600;0,14..32,700;0,14..32,800&display=swap" rel="stylesheet" />
      </head>
      <body className="flex h-screen overflow-hidden bg-background text-foreground font-sans antialiased">
        <Sidebar />
        <main className="flex-1 overflow-y-auto">
          {children}
        </main>
      </body>
    </html>
  )
}
