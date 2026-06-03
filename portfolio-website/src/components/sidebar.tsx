'use client'

import { usePathname } from 'next/navigation'
import Link from 'next/link'
import { cn } from '@/lib/utils'
import { CATEGORY_GROUPS } from '@/lib/projects'
import { ScrollArea } from '@/components/ui/scroll-area'
import { LayoutDashboard, Brain } from 'lucide-react'

export function Sidebar() {
  const pathname = usePathname()

  return (
    <aside
      className="hidden lg:flex flex-col w-56 shrink-0 h-screen sticky top-0"
      style={{ background: '#111827', borderRight: '1px solid #1f2937' }}
    >
      {/* Logo */}
      <div
        className="flex items-center gap-3 px-4 h-14"
        style={{ borderBottom: '1px solid #1f2937' }}
      >
        <div
          className="flex items-center justify-center w-8 h-8 rounded-lg shrink-0"
          style={{ background: '#1d4ed8' }}
        >
          <Brain className="w-4 h-4 text-white" />
        </div>
        <div className="min-w-0">
          <div className="text-sm font-semibold text-white leading-none">AI Portfolio</div>
          <div className="text-[10px] mt-0.5 truncate" style={{ color: '#9ca3af' }}>
            Oluwafemi Adeyemi
          </div>
        </div>
      </div>

      <ScrollArea className="flex-1 px-2 py-2">
        {/* Overview link */}
        <Link
          href="/"
          className={cn(
            'flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm font-medium transition-colors mb-1',
            pathname === '/'
              ? 'text-white'
              : 'hover:text-white'
          )}
          style={
            pathname === '/'
              ? { background: '#1f2937', color: '#fff' }
              : { color: '#9ca3af' }
          }
        >
          <LayoutDashboard className="w-4 h-4 shrink-0" />
          Portfolio Overview
        </Link>

        <div style={{ height: 1, background: '#1f2937', margin: '8px 4px' }} />

        {/* Project groups */}
        {CATEGORY_GROUPS.map(group => (
          <div key={group.label} className="mb-3">
            <div
              className="px-3 mb-1 text-[10px] font-semibold uppercase tracking-wider"
              style={{ color: '#6b7280' }}
            >
              {group.label}
            </div>
            <div className="space-y-0.5">
              {group.projects.map(proj => {
                const active = pathname === `/projects/${proj.id}`
                return (
                  <Link
                    key={proj.id}
                    href={`/projects/${proj.id}`}
                    className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm transition-colors"
                    style={
                      active
                        ? { background: '#1f2937', color: '#fff', fontWeight: 500 }
                        : { color: '#9ca3af' }
                    }
                    onMouseEnter={e => {
                      if (!active) (e.currentTarget as HTMLElement).style.color = '#e5e7eb'
                    }}
                    onMouseLeave={e => {
                      if (!active) (e.currentTarget as HTMLElement).style.color = '#9ca3af'
                    }}
                  >
                    <span className="text-sm leading-none shrink-0">{proj.icon}</span>
                    <span className="truncate">{proj.short}</span>
                    {active && (
                      <span
                        className="ml-auto w-1.5 h-1.5 rounded-full shrink-0"
                        style={{ background: proj.color }}
                      />
                    )}
                  </Link>
                )
              })}
            </div>
          </div>
        ))}
      </ScrollArea>

      {/* Footer */}
      <div
        className="px-4 py-3"
        style={{ borderTop: '1px solid #1f2937' }}
      >
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse shrink-0" />
          <span className="text-[11px]" style={{ color: '#6b7280' }}>
            17 projects · Llama 3.2 local
          </span>
        </div>
      </div>
    </aside>
  )
}
