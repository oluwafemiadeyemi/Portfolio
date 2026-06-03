'use client'

import { useState } from 'react'
import Link from 'next/link'
import { PROJECTS, CATEGORIES } from '@/lib/projects'
import { ExternalLink, ArrowRight, GitBranch, Mail, TrendingUp } from 'lucide-react'
import { cn } from '@/lib/utils'

const STATS = [
  { value: '17',       label: 'AI Systems',         sub: 'Production-grade' },
  { value: '24 GB+',   label: 'Real Training Data',  sub: 'No synthetic shortcuts' },
  { value: '60+',      label: 'Trained Models',      sub: 'ML · DL · GenAI · Vision' },
  { value: 'Llama 3.2',label: 'Local LLM',           sub: 'Zero API cost' },
]

export default function Home() {
  const [activeCategory, setActiveCategory] = useState('All')

  const filtered = activeCategory === 'All'
    ? PROJECTS
    : PROJECTS.filter(p => p.category === activeCategory)

  return (
    <div className="min-h-screen bg-white">

      {/* ── Hero ────────────────────────────────────────────────── */}
      <section className="px-8 py-16 border-b border-slate-200 bg-gradient-to-b from-slate-50 to-white">
        <div className="max-w-2xl mx-auto text-center">
          <span className="inline-block px-3 py-1 rounded-full text-xs font-semibold tracking-widest uppercase mb-6"
            style={{ background: '#eff6ff', color: '#1d4ed8', border: '1px solid #bfdbfe' }}>
            MIT Applied AI & Data Science
          </span>
          <h1 className="text-4xl md:text-5xl font-extrabold text-slate-900 mb-3 leading-tight">
            Oluwafemi Adeyemi
          </h1>
          <p className="text-lg font-medium text-slate-600 mb-2">
            Applied AI Engineer & Data Scientist
          </p>
          <p className="text-slate-500 mb-10 leading-relaxed max-w-xl mx-auto">
            17 production-grade AI systems — healthcare, finance, retail, HR, entertainment.
            Each ships with a REST API, live demo, and full explainability.
          </p>

          {/* Stats row */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-10">
            {STATS.map(s => (
              <div key={s.label}
                className="rounded-xl p-4 text-center border border-slate-200 bg-white shadow-sm">
                <div className="text-2xl font-bold text-blue-600 mb-0.5">{s.value}</div>
                <div className="text-xs font-semibold text-slate-700">{s.label}</div>
                <div className="text-[11px] text-slate-400 mt-0.5">{s.sub}</div>
              </div>
            ))}
          </div>

          {/* CTAs */}
          <div className="flex gap-3 justify-center flex-wrap">
            <a href="https://github.com/oluwafemiadeyemi/Portfolio"
              target="_blank" rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium border border-slate-300 text-slate-700 bg-white hover:bg-slate-50 transition-colors">
              <GitBranch className="w-4 h-4" /> GitHub
            </a>
            <a href="mailto:femi@phoxta.com"
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 transition-colors">
              <Mail className="w-4 h-4" /> Get in Touch
            </a>
          </div>
        </div>
      </section>

      {/* ── Filter + Grid ────────────────────────────────────────── */}
      <section className="px-6 py-8 max-w-7xl mx-auto">

        {/* Category filter pills */}
        <div className="flex gap-2 flex-wrap mb-8">
          {CATEGORIES.map(cat => (
            <button
              key={cat}
              onClick={() => setActiveCategory(cat)}
              className={cn(
                'px-3 py-1.5 rounded-full text-xs font-medium border transition-all',
                activeCategory === cat
                  ? 'bg-blue-600 text-white border-blue-600 shadow-sm'
                  : 'bg-white text-slate-600 border-slate-300 hover:border-blue-400 hover:text-blue-600'
              )}
            >
              {cat}
              {cat !== 'All' && (
                <span className={cn(
                  'ml-1.5',
                  activeCategory === cat ? 'text-blue-200' : 'text-slate-400'
                )}>
                  {PROJECTS.filter(p => p.category === cat).length}
                </span>
              )}
            </button>
          ))}
        </div>

        {/* Project grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5">
          {filtered.map(project => (
            <ProjectCard key={project.id} project={project} />
          ))}
        </div>
      </section>

      {/* ── Footer ───────────────────────────────────────────────── */}
      <footer className="border-t border-slate-200 bg-slate-50 px-8 py-8 text-center mt-8">
        <p className="text-sm text-slate-500 mb-4">
          Built by <strong className="text-slate-800">Oluwafemi Adeyemi</strong> · MIT Applied AI & Data Science
        </p>
        <div className="flex gap-6 justify-center">
          <a href="mailto:femi@phoxta.com" className="text-sm text-blue-600 hover:underline">femi@phoxta.com</a>
          <a href="https://github.com/oluwafemiadeyemi" target="_blank" rel="noopener noreferrer" className="text-sm text-blue-600 hover:underline">GitHub</a>
          <a href="https://www.linkedin.com/in/oluwafemiadeyemi" target="_blank" rel="noopener noreferrer" className="text-sm text-blue-600 hover:underline">LinkedIn</a>
        </div>
      </footer>
    </div>
  )
}

// ── Project Card ──────────────────────────────────────────────────────────────

function ProjectCard({ project }: { project: typeof PROJECTS[0] }) {
  return (
    <div className="group flex flex-col bg-white rounded-xl border border-slate-200 overflow-hidden hover:shadow-md hover:-translate-y-0.5 transition-all duration-200">

      {/* Colour accent top bar */}
      <div className="h-1 w-full shrink-0" style={{ background: project.color }} />

      <div className="flex flex-col flex-1 p-5 gap-4">

        {/* Header */}
        <div className="flex items-start gap-3">
          <div
            className="flex items-center justify-center w-10 h-10 rounded-xl text-xl shrink-0"
            style={{ background: project.color + '18' }}
          >
            {project.icon}
          </div>
          <div className="min-w-0">
            <div
              className="text-[10px] font-semibold uppercase tracking-wider mb-1"
              style={{ color: project.color }}
            >
              P{String(project.num).padStart(2, '0')} · {project.category}
            </div>
            <h2 className="text-[15px] font-semibold text-slate-900 leading-snug line-clamp-2">
              {project.name}
            </h2>
          </div>
        </div>

        {/* Description */}
        <p className="text-[13px] text-slate-500 leading-relaxed line-clamp-2 flex-1">
          {project.description}
        </p>

        {/* KPI metrics */}
        <div className="grid grid-cols-2 gap-2">
          {project.metrics.slice(0, 4).map(m => (
            <div key={m.label} className="rounded-lg px-3 py-2 bg-slate-50 border border-slate-100">
              <div className="text-[15px] font-bold" style={{ color: project.color }}>
                {m.value}
              </div>
              <div className="text-[10px] text-slate-500 mt-0.5">{m.label}</div>
            </div>
          ))}
        </div>

        {/* Tech stack badges */}
        <div className="flex flex-wrap gap-1.5">
          {project.stack.map(t => (
            <span key={t}
              className="px-2 py-0.5 rounded-md text-[11px] font-medium bg-slate-100 text-slate-600 border border-slate-200">
              {t}
            </span>
          ))}
        </div>

        {/* Target buyers */}
        <div className="flex items-center gap-1.5 flex-wrap">
          <span className="text-[10px] font-medium text-slate-400">Buyers:</span>
          {project.buyers.map(b => (
            <span key={b} className="text-[11px] font-medium text-slate-600">{b}</span>
          ))}
        </div>

        {/* Divider */}
        <div className="h-px bg-slate-100" />

        {/* Action buttons */}
        <div className="flex gap-2">
          <Link href={`/projects/${project.id}`}
            className="flex-1 inline-flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg text-xs font-semibold text-white transition-opacity hover:opacity-90"
            style={{ background: project.color }}>
            Open Dashboard <ArrowRight className="w-3 h-3" />
          </Link>
          {project.demoUrl && (
            <a href={project.demoUrl} target="_blank" rel="noopener noreferrer"
              className="inline-flex items-center justify-center w-9 h-9 rounded-lg border border-slate-200 text-slate-500 hover:text-slate-800 hover:bg-slate-50 transition-colors">
              <ExternalLink className="w-3.5 h-3.5" />
            </a>
          )}
          {project.githubUrl && (
            <a href={project.githubUrl} target="_blank" rel="noopener noreferrer"
              className="inline-flex items-center justify-center w-9 h-9 rounded-lg border border-slate-200 text-slate-500 hover:text-slate-800 hover:bg-slate-50 transition-colors">
              <GitBranch className="w-3.5 h-3.5" />
            </a>
          )}
        </div>
      </div>
    </div>
  )
}
