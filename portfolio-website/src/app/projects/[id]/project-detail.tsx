'use client'

import Link from 'next/link'
import type { Project } from '@/lib/projects'
import { PROJECTS } from '@/lib/projects'
import { ExternalLink, ArrowLeft, Cpu, Users, TrendingUp,
         ChevronRight, Play, GitBranch, Database, BarChart3, Info } from 'lucide-react'

export function ProjectDetail({ project }: { project: Project }) {
  const idx  = PROJECTS.findIndex(p => p.id === project.id)
  const prev = PROJECTS[idx - 1]
  const next = PROJECTS[idx + 1]

  return (
    <div className="min-h-screen bg-white">

      {/* ── Sticky topbar ──────────────────────────────────────────── */}
      <div className="sticky top-0 z-10 bg-white border-b border-slate-200 px-6 h-14 flex items-center justify-between shadow-sm">
        <div className="flex items-center gap-2 text-sm">
          <Link href="/"
            className="inline-flex items-center gap-1.5 text-slate-500 hover:text-slate-800 transition-colors font-medium">
            <ArrowLeft className="w-4 h-4" /> Portfolio
          </Link>
          <ChevronRight className="w-3.5 h-3.5 text-slate-300" />
          <span className="text-slate-800 font-medium truncate max-w-[220px]">{project.name}</span>
        </div>
        <div className="flex gap-2">
          {project.demoUrl && (
            <a href={project.demoUrl} target="_blank" rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border border-slate-300 text-slate-600 hover:bg-slate-50 transition-colors">
              <ExternalLink className="w-3 h-3" /> Live Demo
            </a>
          )}
          {project.githubUrl && (
            <a href={project.githubUrl} target="_blank" rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border border-slate-300 text-slate-600 hover:bg-slate-50 transition-colors">
              <GitBranch className="w-3 h-3" /> Source
            </a>
          )}
        </div>
      </div>

      <div className="max-w-5xl mx-auto px-6 py-10 space-y-8">

        {/* ── Hero ───────────────────────────────────────────────────── */}
        <div className="flex items-start gap-5">
          <div
            className="flex items-center justify-center w-16 h-16 rounded-2xl text-3xl shrink-0"
            style={{ background: project.color + '18' }}
          >
            {project.icon}
          </div>
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2 flex-wrap mb-2">
              <span
                className="inline-block px-2.5 py-0.5 rounded-full text-[10px] font-semibold uppercase tracking-wider border"
                style={{ color: project.color, borderColor: project.color + '40', background: project.color + '0D' }}
              >
                P{String(project.num).padStart(2, '0')} · {project.category}
              </span>
              <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[10px] font-medium bg-slate-100 text-slate-600 border border-slate-200">
                <Database className="w-2.5 h-2.5" /> {project.dataset}
              </span>
            </div>
            <h1 className="text-2xl md:text-3xl font-bold text-slate-900 mb-2">{project.name}</h1>
            <p className="text-slate-500 leading-relaxed max-w-2xl">{project.description}</p>
            <div className="flex items-center gap-2 mt-3 flex-wrap">
              <span className="text-xs text-slate-400 font-medium">Target buyers:</span>
              {project.buyers.map(b => (
                <span key={b}
                  className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs font-medium bg-slate-100 text-slate-700 border border-slate-200">
                  <Users className="w-2.5 h-2.5" /> {b}
                </span>
              ))}
            </div>
          </div>
        </div>

        {/* ── KPI cards ─────────────────────────────────────────────── */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {project.metrics.map((m, i) => (
            <div key={i} className="rounded-xl p-4 bg-white border border-slate-200 shadow-sm">
              <div className="text-2xl font-bold mb-1" style={{ color: project.color }}>{m.value}</div>
              <div className="text-xs text-slate-500">{m.label}</div>
              {m.delta && (
                <div className="text-xs text-emerald-600 mt-1 flex items-center gap-0.5 font-medium">
                  <TrendingUp className="w-3 h-3" /> {m.delta}
                </div>
              )}
            </div>
          ))}
        </div>

        {/* ── Tabs ──────────────────────────────────────────────────── */}
        <ProjectTabs project={project} />

        {/* ── Prev / Next navigation ─────────────────────────────── */}
        <div className="border-t border-slate-200 pt-6 flex items-center justify-between">
          {prev ? (
            <Link href={`/projects/${prev.id}`}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium text-slate-600 hover:text-slate-900 hover:bg-slate-50 border border-slate-200 transition-colors">
              <ArrowLeft className="w-4 h-4" />
              <span>{prev.icon} {prev.short}</span>
            </Link>
          ) : <div />}

          <span className="text-xs font-medium text-slate-400 tabular-nums">
            {project.num} / {PROJECTS.length}
          </span>

          {next ? (
            <Link href={`/projects/${next.id}`}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium text-slate-600 hover:text-slate-900 hover:bg-slate-50 border border-slate-200 transition-colors">
              <span>{next.icon} {next.short}</span>
              <ChevronRight className="w-4 h-4" />
            </Link>
          ) : <div />}
        </div>
      </div>
    </div>
  )
}

// ── Tab component (no Radix dep — simple state) ───────────────────────────────

function ProjectTabs({ project }: { project: Project }) {
  const tabs = [
    { id: 'overview', label: 'Overview',          icon: BarChart3 },
    { id: 'demo',     label: 'Demo & APIs',        icon: Play },
    { id: 'details',  label: 'Technical Details',  icon: Info },
  ] as const

  type TabId = typeof tabs[number]['id']
  const [active, setActive] = useState<TabId>('overview')

  return (
    <div>
      {/* Tab bar */}
      <div className="flex gap-1 p-1 rounded-lg bg-slate-100 w-fit mb-6">
        {tabs.map(t => {
          const Icon = t.icon
          return (
            <button
              key={t.id}
              onClick={() => setActive(t.id)}
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-all ${
                active === t.id
                  ? 'bg-white text-slate-900 shadow-sm'
                  : 'text-slate-500 hover:text-slate-700'
              }`}
            >
              <Icon className="w-3.5 h-3.5" /> {t.label}
            </button>
          )
        })}
      </div>

      {/* Overview */}
      {active === 'overview' && (
        <div className="grid md:grid-cols-2 gap-5">
          <div className="rounded-xl border border-slate-200 bg-white p-5">
            <h3 className="text-sm font-semibold text-slate-800 mb-3">Technology Stack</h3>
            <div className="flex flex-wrap gap-2">
              {project.stack.map(t => (
                <div key={t} className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-50 border border-slate-200 text-xs font-medium text-slate-700">
                  <Cpu className="w-3 h-3 text-blue-500" /> {t}
                </div>
              ))}
            </div>
          </div>

          <div className="rounded-xl border border-slate-200 bg-white p-5">
            <h3 className="text-sm font-semibold text-slate-800 mb-3">Key Metrics</h3>
            <div className="space-y-2.5">
              {project.metrics.map((m, i) => (
                <div key={i} className="flex items-center justify-between">
                  <span className="text-xs text-slate-500">{m.label}</span>
                  <span className="text-sm font-bold" style={{ color: project.color }}>{m.value}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="rounded-xl border border-slate-200 bg-white p-5 md:col-span-2">
            <h3 className="text-sm font-semibold text-slate-800 mb-3">Project Description</h3>
            <p className="text-sm text-slate-600 leading-relaxed">{project.description}</p>
            <div className="border-t border-slate-100 mt-4 pt-4 grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              {[
                { label: 'Dataset',   value: project.dataset },
                { label: 'Domain',    value: project.category },
                { label: 'API Port',  value: `:${project.apiPort}` },
                { label: 'Dash Port', value: `:${project.dashPort}` },
              ].map(r => (
                <div key={r.label}>
                  <div className="text-[10px] uppercase tracking-wide text-slate-400 font-medium mb-1">{r.label}</div>
                  <div className="text-xs font-semibold text-slate-700">{r.value}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Demo & APIs */}
      {active === 'demo' && (
        <div className="rounded-xl border border-slate-200 bg-white p-8 text-center">
          <div className="text-5xl mb-4">{project.icon}</div>
          <h3 className="text-xl font-semibold text-slate-900 mb-2">Interactive Demos Available</h3>
          <p className="text-sm text-slate-500 max-w-md mx-auto mb-8">
            Run locally with <code className="bg-slate-100 px-1.5 py-0.5 rounded text-xs text-slate-700 font-mono">streamlit run dashboard/app.py</code>, or use the hosted Hugging Face demo.
          </p>
          <div className="flex gap-3 justify-center flex-wrap">
            {project.demoUrl && (
              <a href={project.demoUrl} target="_blank" rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-semibold text-white transition-opacity hover:opacity-90"
                style={{ background: project.color }}>
                <ExternalLink className="w-4 h-4" /> Hosted Demo
              </a>
            )}
            <a href={`http://localhost:${project.dashPort}`} target="_blank" rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-semibold text-slate-700 border border-slate-300 hover:bg-slate-50 transition-colors">
              <Play className="w-4 h-4" /> Local :{project.dashPort}
            </a>
            <a href={`http://localhost:${project.apiPort}/docs`} target="_blank" rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-semibold text-slate-700 border border-slate-300 hover:bg-slate-50 transition-colors">
              <Cpu className="w-4 h-4" /> API Docs :{project.apiPort}
            </a>
          </div>
          <div className="mt-6 text-xs text-slate-400">
            Or view in the{' '}
            <a href="http://localhost:8600" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline font-medium">
              Unified Streamlit Dashboard (:8600)
            </a>
          </div>
        </div>
      )}

      {/* Technical Details */}
      {active === 'details' && (
        <div className="grid md:grid-cols-2 gap-5">
          <div className="rounded-xl border border-slate-200 bg-white p-5">
            <h3 className="text-sm font-semibold text-slate-800 mb-3">Target Buyers</h3>
            <div className="space-y-2">
              {project.buyers.map(b => (
                <div key={b} className="flex items-center gap-2 text-sm text-slate-700">
                  <span className="w-2 h-2 rounded-full shrink-0" style={{ background: project.color }} />
                  {b}
                </div>
              ))}
            </div>
          </div>

          <div className="rounded-xl border border-slate-200 bg-white p-5">
            <h3 className="text-sm font-semibold text-slate-800 mb-3">Deployment</h3>
            <div className="space-y-2.5">
              {[
                { label: 'FastAPI',    value: `:${project.apiPort}` },
                { label: 'Streamlit',  value: `:${project.dashPort}` },
                { label: 'Hosted',     value: 'HF Spaces' },
                { label: 'Container',  value: 'Dockerfile included' },
                { label: 'Model',      value: 'joblib / ONNX / PyTorch' },
              ].map(r => (
                <div key={r.label} className="flex justify-between items-center text-sm">
                  <span className="text-slate-500 text-xs">{r.label}</span>
                  <code className="text-xs bg-slate-100 px-2 py-0.5 rounded font-mono text-slate-700">{r.value}</code>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

// useState import
import { useState } from 'react'
