// Server component — handles static generation; renders the client component
import { notFound } from 'next/navigation'
import { PROJECT_MAP, PROJECTS } from '@/lib/projects'
import { ProjectDetail } from './project-detail'

export async function generateStaticParams() {
  return PROJECTS.map(p => ({ id: p.id }))
}

export default function ProjectPage({ params }: { params: { id: string } }) {
  const project = PROJECT_MAP[params.id]
  if (!project) notFound()
  return <ProjectDetail project={project} />
}
