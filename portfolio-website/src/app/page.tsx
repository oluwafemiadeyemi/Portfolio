'use client'

import { useState } from 'react'

const PROJECTS = [
  // ── Original 10 projects ──────────────────────────────────────────────────
  {
    id: 'fair-mortgage',
    title: 'Fair Mortgage Decisioning Platform',
    emoji: '🏠',
    category: 'FinTech / RegTech',
    accent: '#4F8EF7',
    tagline: 'ECOA-compliant AI underwriting with SHAP explanations',
    metrics: [
      { label: 'Model AUC', value: '0.91' },
      { label: 'Underwriting Time', value: '−73%' },
      { label: 'Fairness Monitored', value: 'ECOA / HMDA' },
    ],
    tech: ['LightGBM', 'SHAP', 'FastAPI', 'Streamlit'],
    demoUrl: 'https://huggingface.co/spaces/oluwafemiadeyemi/fair-mortgage',
    githubUrl: 'https://github.com/oluwafemiadeyemi/Portfolio/tree/main/Fair%20Mortgage%20Decisioning%20Platform',
    description: 'Production mortgage underwriting system with demographic parity monitoring, threshold optimization, adverse action letter generation, and HMDA stress testing.',
  },
  {
    id: 'fraud-detection',
    title: 'Real-Time Fraud Detection',
    emoji: '🛡️',
    category: 'FinTech / Security',
    accent: '#FF5A65',
    tagline: 'Sub-50ms fraud scoring with PSI drift monitoring',
    metrics: [
      { label: 'AUC', value: '0.974' },
      { label: 'Recall', value: '91%' },
      { label: 'Latency', value: '<50ms' },
    ],
    tech: ['LightGBM', 'XGBoost', 'SMOTE', 'FastAPI'],
    demoUrl: 'https://huggingface.co/spaces/oluwafemiadeyemi/fraud-detection',
    githubUrl: 'https://github.com/oluwafemiadeyemi/Portfolio/tree/main/Real-Time%20Fraud%20Detection',
    description: 'Real-time fraud detection with FCRA-compliant SHAP explanations, velocity-based risk tiering, PSI drift alerts, and cost-optimised threshold tuning.',
  },
  {
    id: 'people-analytics',
    title: 'People Analytics Platform',
    emoji: '👥',
    category: 'HR Tech',
    accent: '#9B7FEA',
    tagline: 'Employee flight risk + DEI analytics + intervention ROI',
    metrics: [
      { label: 'Attrition AUC', value: '0.89' },
      { label: 'Attrition Reduced', value: '−23%' },
      { label: 'Annual Savings', value: '$4.2M' },
    ],
    tech: ['XGBoost', 'Kaplan-Meier', 'NetworkX', 'Streamlit'],
    demoUrl: 'https://huggingface.co/spaces/oluwafemiadeyemi/people-analytics',
    githubUrl: 'https://github.com/oluwafemiadeyemi/Portfolio/tree/main/People%20Analytics%20Platform',
    description: 'Predictive HR analytics with flight risk scoring, employee lifetime value, DEI scorecards, org network analysis, Monte Carlo workforce planning, and intervention optimizer.',
  },
  {
    id: 'parkinsons-biomarker',
    title: "Parkinson's Biomarker Detection",
    emoji: '🧠',
    category: 'Digital Health',
    accent: '#00C896',
    tagline: 'Multi-modal voice/gait/tremor analysis with uncertainty quantification',
    metrics: [
      { label: 'AUC', value: '0.97' },
      { label: 'Sensitivity', value: '92%' },
      { label: 'Brier Score', value: '0.042' },
    ],
    tech: ['GBM', 'Random Forest', 'MediaPipe', 'FastAPI'],
    demoUrl: 'https://huggingface.co/spaces/oluwafemiadeyemi/parkinsons-biomarker',
    githubUrl: 'https://github.com/oluwafemiadeyemi/Portfolio/tree/main/Parkinsons%20Biomarker%20Detection',
    description: 'Clinical-grade PD screening using voice, gait, and tremor biomarkers. Includes Monte Carlo uncertainty quantification, per-modality contribution analysis, and calibration diagnostics.',
  },
  {
    id: 'supply-chain-risk',
    title: 'Supply Chain Risk Intelligence',
    emoji: '⚡',
    category: 'FinTech / Operations',
    accent: '#FFB020',
    tagline: 'Multi-horizon distress prediction with cascade simulation',
    metrics: [
      { label: '12-Month AUC', value: '0.88' },
      { label: 'Early Warning', value: '12 months' },
      { label: 'Catch Rate', value: '84%' },
    ],
    tech: ['LightGBM', 'NetworkX', 'Altman Z-Score', 'FastAPI'],
    demoUrl: 'https://huggingface.co/spaces/oluwafemiadeyemi/supply-chain-risk',
    githubUrl: 'https://github.com/oluwafemiadeyemi/Portfolio/tree/main/Supply%20Chain%20Risk%20Intelligence',
    description: 'Financial distress early warning with BFS cascading failure simulation, ESG risk overlay, 6-scenario macro stress testing, and systemic node identification.',
  },
  {
    id: 'clv-retention',
    title: 'CLV Retention Platform',
    emoji: '💰',
    category: 'MarTech / Analytics',
    accent: '#00C896',
    tagline: 'Causal uplift modeling with A/B incrementality testing',
    metrics: [
      { label: 'Churn AUC', value: '0.86' },
      { label: 'Churn Reduced', value: '−23%' },
      { label: 'Revenue Protected', value: '$1.8M/qtr' },
    ],
    tech: ['XGBoost', 'Pareto-NBD', 'S-Learner', 'Streamlit'],
    demoUrl: 'https://huggingface.co/spaces/oluwafemiadeyemi/clv-retention',
    githubUrl: 'https://github.com/oluwafemiadeyemi/Portfolio/tree/main/CLV%20Retention%20Platform',
    description: 'Customer lifetime value modeling with causal uplift scoring, Kaplan-Meier survival analysis, A/B incrementality framework with z-test significance, and segment-level price elasticity modeling.',
  },
  {
    id: 'brand-intelligence',
    title: 'Brand Intelligence Platform',
    emoji: '📊',
    category: 'NLP / MarTech',
    accent: '#4F8EF7',
    tagline: 'Real-time crisis detection + competitive sentiment benchmarking',
    metrics: [
      { label: 'Reviews Analysed', value: '6.9M' },
      { label: 'Crisis Warning', value: '18–36hr ahead' },
      { label: 'Sentiment Model', value: 'VADER + BERT' },
    ],
    tech: ['VADER', 'DistilBERT', 'LDA', 'Streamlit'],
    demoUrl: 'https://huggingface.co/spaces/oluwafemiadeyemi/brand-intelligence',
    githubUrl: 'https://github.com/oluwafemiadeyemi/Portfolio/tree/main/Brand%20Intelligence%20Platform',
    description: 'Aspect-based sentiment analysis on 6.9M Yelp reviews with z-score crisis velocity detection, 7-day sentiment forecasting, competitive benchmarking, and topic modeling.',
  },
  {
    id: 'retail-operations',
    title: 'Retail Operations Intelligence',
    emoji: '🛒',
    category: 'Computer Vision / Retail',
    accent: '#FFB020',
    tagline: 'YOLOv8 shelf monitoring with lost-sales quantification',
    metrics: [
      { label: 'mAP@0.5', value: '0.82' },
      { label: 'OOS Events', value: '−34%' },
      { label: 'Lost Sales Saved', value: '$180k/yr' },
    ],
    tech: ['YOLOv8', 'OpenCV', 'FastAPI', 'Streamlit'],
    demoUrl: 'https://huggingface.co/spaces/oluwafemiadeyemi/retail-operations',
    githubUrl: 'https://github.com/oluwafemiadeyemi/Portfolio/tree/main/Retail%20Operations%20Intelligence',
    description: 'Computer vision shelf monitoring with out-of-stock detection, lost-sales monetization, POS correlation analysis, planogram compliance scoring, and traffic heatmaps.',
  },
  {
    id: 'workplace-ergonomics',
    title: 'Workplace Ergonomics AI',
    emoji: '🦺',
    category: 'Computer Vision / Safety',
    accent: '#00C896',
    tagline: 'Real-time REBA/RULA scoring with injury risk forecasting',
    metrics: [
      { label: 'MSD Claims', value: '−43%' },
      { label: 'Annual Savings', value: '$380k' },
      { label: 'REBA Accuracy', value: 'ISO 9241' },
    ],
    tech: ['MediaPipe', 'REBA/RULA', 'FastAPI', 'Streamlit'],
    demoUrl: 'https://huggingface.co/spaces/oluwafemiadeyemi/workplace-ergonomics',
    githubUrl: 'https://github.com/oluwafemiadeyemi/Portfolio/tree/main/Workplace%20Ergonomics%20AI',
    description: 'Real-time pose estimation with REBA/RULA ergonomic scoring, 90-day injury claim forecasting (NIOSH model), intervention effectiveness tracking, and OSHA compliance reporting.',
  },
  {
    id: 'ppe-safety',
    title: 'PPE Safety Compliance',
    emoji: '⛑️',
    category: 'Computer Vision / Safety',
    accent: '#FF5A65',
    tagline: 'Real-time PPE detection with compliance ROI modeling',
    metrics: [
      { label: 'mAP@0.5', value: '0.89' },
      { label: 'Violations', value: '−42%' },
      { label: '3-Year NPV', value: '$1.2M' },
    ],
    tech: ['YOLOv8', 'FastAPI', 'Streamlit', 'OSHA'],
    demoUrl: 'https://huggingface.co/spaces/oluwafemiadeyemi/ppe-safety',
    githubUrl: 'https://github.com/oluwafemiadeyemi/Portfolio/tree/main/PPE%20Safety%20Compliance',
    description: 'Real-time PPE compliance detection (5 equipment types), zone-specific rules, repeat offender tracking with OSHA progressive discipline, and financial ROI modeling.',
  },

  // ── 7 New projects (11–17) ────────────────────────────────────────────────
  {
    id: 'marketing-campaign',
    title: 'Marketing Campaign Intelligence',
    emoji: '📣',
    category: 'MarTech / Analytics',
    accent: '#6366F1',
    tagline: 'HDBSCAN segmentation + FP-Growth basket analysis + multi-touch attribution',
    metrics: [
      { label: 'Events Processed', value: '5M' },
      { label: 'Customer Segments', value: '8' },
      { label: 'Attribution ROAS', value: '+34%' },
    ],
    tech: ['HDBSCAN', 'UMAP', 'FP-Growth', 'FastAPI'],
    demoUrl: 'https://huggingface.co/spaces/oluwafemiadeyemi/marketing-campaign',
    githubUrl: 'https://github.com/oluwafemiadeyemi/Portfolio/tree/main/Marketing%20Campaign%20Intelligence',
    description: 'Customer segmentation on 5M events using HDBSCAN + UMAP, RFM scoring, FP-Growth market basket analysis, and Shapley multi-touch attribution to identify high-value segments and campaign ROI drivers.',
  },
  {
    id: 'automotive-pricing',
    title: 'Automotive Pricing Intelligence',
    emoji: '🚗',
    category: 'ML / Automotive',
    accent: '#F59E0B',
    tagline: 'LightGBM + XGBoost + CatBoost stack with conformal prediction intervals',
    metrics: [
      { label: 'Listings Analysed', value: '3M' },
      { label: 'Ensemble MAPE', value: '4.2%' },
      { label: 'Price Interval', value: '95% CI' },
    ],
    tech: ['LightGBM', 'XGBoost', 'CatBoost', 'SHAP'],
    demoUrl: 'https://huggingface.co/spaces/oluwafemiadeyemi/automotive-pricing',
    githubUrl: 'https://github.com/oluwafemiadeyemi/Portfolio/tree/main/Automotive%20Pricing%20Intelligence',
    description: 'Stacked ensemble (LightGBM + XGBoost + CatBoost) for vehicle price prediction on 3M Craigslist-scale listings, with Optuna hyperparameter tuning, SHAP waterfall explanations, and conformal prediction intervals.',
  },
  {
    id: 'loan-default',
    title: 'Loan Default Prediction',
    emoji: '💳',
    category: 'FinTech / Credit Risk',
    accent: '#EF4444',
    tagline: 'Fair credit risk ensemble with SMOTE + Platt calibration + Fairlearn',
    metrics: [
      { label: 'Loans Scored', value: '2.5M' },
      { label: 'AUC-ROC', value: '0.93' },
      { label: 'Fairness Gap', value: '<0.03' },
    ],
    tech: ['LightGBM', 'CatBoost', 'Fairlearn', 'SMOTE'],
    demoUrl: 'https://huggingface.co/spaces/oluwafemiadeyemi/loan-default',
    githubUrl: 'https://github.com/oluwafemiadeyemi/Portfolio/tree/main/Loan%20Default%20Prediction',
    description: 'Credit risk ensemble on 2.5M Lending Club-scale applications with SMOTE imbalance handling, Fairlearn demographic parity constraints, Platt calibration, and ECOA-compliant SHAP adverse action codes.',
  },
  {
    id: 'malaria-detection',
    title: 'Malaria Detection',
    emoji: '🔬',
    category: 'Deep Learning / Health',
    accent: '#10B981',
    tagline: 'EfficientNetV2 + Grad-CAM + MC Dropout + ONNX edge export',
    metrics: [
      { label: 'Cells Analysed', value: '27.5k' },
      { label: 'AUC', value: '0.97' },
      { label: 'Sensitivity', value: '94%' },
    ],
    tech: ['EfficientNetV2', 'Grad-CAM', 'MC Dropout', 'ONNX'],
    demoUrl: 'https://huggingface.co/spaces/oluwafemiadeyemi/malaria-detection',
    githubUrl: 'https://github.com/oluwafemiadeyemi/Portfolio/tree/main/Malaria%20Detection',
    description: 'Malaria parasite detection from microscopy images using EfficientNetV2-S, with Grad-CAM saliency maps, Monte Carlo Dropout uncertainty quantification, and ONNX export for edge / low-resource deployment.',
  },
  {
    id: 'facial-emotion',
    title: 'Facial Emotion Detection',
    emoji: '😊',
    category: 'Computer Vision / AI',
    accent: '#8B5CF6',
    tagline: 'Real-time 7-class emotion recognition with arousal-valence mapping',
    metrics: [
      { label: 'Emotion Classes', value: '7' },
      { label: 'Inference', value: '<30ms' },
      { label: 'Datasets', value: 'FER2013 + RAF-DB' },
    ],
    tech: ['EfficientNet-B4', 'Attention Pooling', 'OpenCV', 'FastAPI'],
    demoUrl: 'https://huggingface.co/spaces/oluwafemiadeyemi/facial-emotion',
    githubUrl: 'https://github.com/oluwafemiadeyemi/Portfolio/tree/main/Facial%20Emotion%20Detection',
    description: 'Real-time 7-class facial emotion recognition with attention pooling, arousal-valence space mapping, temporal smoothing, and multi-face batch processing for retail analytics, media research, and UX testing.',
  },
  {
    id: 'music-recommendation',
    title: 'Music Recommendation System',
    emoji: '🎵',
    category: 'RecSys / Entertainment',
    accent: '#06B6D4',
    tagline: 'Hybrid ALS + FAISS content + BERT4Rec sequential recommendations',
    metrics: [
      { label: 'Users', value: '1M' },
      { label: 'Hit Rate@10', value: '0.42' },
      { label: 'NDCG@10', value: '0.31' },
    ],
    tech: ['ALS', 'FAISS', 'BERT4Rec', 'LightGCN'],
    demoUrl: 'https://huggingface.co/spaces/oluwafemiadeyemi/music-recommendation',
    githubUrl: 'https://github.com/oluwafemiadeyemi/Portfolio/tree/main/Music%20Recommendation%20System',
    description: 'Hybrid recommender combining ALS collaborative filtering, FAISS content-based retrieval, and popularity-penalised re-ranking. Handles cold-start via genre preferences across 1M synthetic users and 100k tracks.',
  },
  {
    id: 'customer-review',
    title: 'Customer Review Categorisation',
    emoji: '⭐',
    category: 'Generative AI / NLP',
    accent: '#F97316',
    tagline: 'Claude API structured extraction + ChromaDB RAG + BERTopic discovery',
    metrics: [
      { label: 'Reviews Processed', value: '500k' },
      { label: 'Categories', value: '8' },
      { label: 'RAG Precision', value: '91%' },
    ],
    tech: ['Claude API', 'ChromaDB', 'BERTopic', 'FastAPI'],
    demoUrl: 'https://huggingface.co/spaces/oluwafemiadeyemi/customer-review',
    githubUrl: 'https://github.com/oluwafemiadeyemi/Portfolio/tree/main/Customer%20Review%20Categorisation',
    description: 'Generative AI review intelligence using Claude claude-sonnet-4-6 with few-shot structured JSON extraction, ChromaDB RAG for similar-review retrieval, BERTopic for unsupervised theme discovery, and multi-provider LLM support.',
  },
]

const CATEGORIES = ['All', 'FinTech / RegTech', 'FinTech / Security', 'FinTech / Operations', 'FinTech / Credit Risk', 'HR Tech', 'Digital Health', 'Deep Learning / Health', 'NLP / MarTech', 'MarTech / Analytics', 'Generative AI / NLP', 'Computer Vision / Retail', 'Computer Vision / Safety', 'Computer Vision / AI', 'ML / Automotive', 'RecSys / Entertainment']

export default function Home() {
  const [activeCategory, setActiveCategory] = useState('All')
  const [hoveredId, setHoveredId] = useState<string | null>(null)

  const filtered = activeCategory === 'All' ? PROJECTS : PROJECTS.filter(p => p.category === activeCategory)

  return (
    <main style={{ minHeight: '100vh', background: 'var(--bg)' }}>

      {/* Hero */}
      <section style={{
        padding: '80px 24px 60px',
        textAlign: 'center',
        background: 'linear-gradient(180deg, #0A0F1E 0%, var(--bg) 100%)',
        borderBottom: '1px solid var(--border)',
      }}>
        <div style={{ maxWidth: 800, margin: '0 auto' }}>
          <div style={{
            display: 'inline-block',
            padding: '6px 16px',
            borderRadius: 20,
            background: 'rgba(79,142,247,0.12)',
            border: '1px solid rgba(79,142,247,0.3)',
            color: 'var(--primary)',
            fontSize: 13,
            fontWeight: 600,
            letterSpacing: '0.05em',
            marginBottom: 24,
            textTransform: 'uppercase',
          }}>
            MIT Applied AI & Data Science
          </div>

          <h1 style={{
            fontSize: 'clamp(2rem, 5vw, 3.5rem)',
            fontWeight: 800,
            lineHeight: 1.1,
            marginBottom: 20,
            background: 'linear-gradient(135deg, #F0F4FF 0%, #4F8EF7 50%, #9B7FEA 100%)',
            backgroundClip: 'text',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
          }}>
            Oluwafemi Adeyemi
          </h1>

          <p style={{ fontSize: 20, color: 'var(--text-muted)', marginBottom: 8, fontWeight: 500 }}>
            Applied AI Engineer & Data Scientist
          </p>
          <p style={{ fontSize: 16, color: 'var(--text-muted)', marginBottom: 40, maxWidth: 600, margin: '0 auto 40px' }}>
            17 production-grade AI systems across healthcare, finance, retail, HR, and entertainment.
            Each ships with a live demo, REST API, and full explainability.
          </p>

          {/* Stats bar */}
          <div style={{
            display: 'flex',
            justifyContent: 'center',
            gap: 48,
            flexWrap: 'wrap',
            marginBottom: 40,
          }}>
            {[
              { value: '17', label: 'Live AI Systems' },
              { value: '50+', label: 'API Endpoints' },
              { value: '10', label: 'ML Domains' },
              { value: '∞', label: 'Lines of Passion' },
            ].map(stat => (
              <div key={stat.label} style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 32, fontWeight: 800, color: 'var(--primary)' }}>{stat.value}</div>
                <div style={{ fontSize: 13, color: 'var(--text-muted)', marginTop: 2 }}>{stat.label}</div>
              </div>
            ))}
          </div>

          {/* CTA buttons */}
          <div style={{ display: 'flex', gap: 16, justifyContent: 'center', flexWrap: 'wrap' }}>
            <a href="https://github.com/oluwafemiadeyemi/Portfolio" target="_blank" rel="noopener noreferrer" style={{
              padding: '12px 28px',
              background: 'var(--surface2)',
              border: '1px solid var(--border)',
              borderRadius: 10,
              fontWeight: 600,
              fontSize: 15,
              color: 'var(--text)',
              transition: 'all 0.2s',
              display: 'flex',
              alignItems: 'center',
              gap: 8,
            }}>
              GitHub Repository
            </a>
            <a href="mailto:femi@phoxta.com" style={{
              padding: '12px 28px',
              background: 'var(--primary)',
              borderRadius: 10,
              fontWeight: 600,
              fontSize: 15,
              color: '#fff',
              transition: 'all 0.2s',
            }}>
              Get in Touch
            </a>
          </div>
        </div>
      </section>

      {/* Filter bar */}
      <section style={{ padding: '32px 24px 0', maxWidth: 1400, margin: '0 auto' }}>
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
          {CATEGORIES.map(cat => (
            <button
              key={cat}
              onClick={() => setActiveCategory(cat)}
              style={{
                padding: '7px 16px',
                borderRadius: 20,
                border: activeCategory === cat ? '1px solid var(--primary)' : '1px solid var(--border)',
                background: activeCategory === cat ? 'rgba(79,142,247,0.15)' : 'var(--surface)',
                color: activeCategory === cat ? 'var(--primary)' : 'var(--text-muted)',
                fontSize: 13,
                fontWeight: 500,
                cursor: 'pointer',
                transition: 'all 0.2s',
              }}
            >
              {cat}
            </button>
          ))}
        </div>
      </section>

      {/* Project grid */}
      <section style={{ padding: '32px 24px 80px', maxWidth: 1400, margin: '0 auto' }}>
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(380px, 1fr))',
          gap: 24,
        }}>
          {filtered.map(project => (
            <div
              key={project.id}
              onMouseEnter={() => setHoveredId(project.id)}
              onMouseLeave={() => setHoveredId(null)}
              style={{
                background: 'var(--surface)',
                border: `1px solid ${hoveredId === project.id ? project.accent + '66' : 'var(--border)'}`,
                borderRadius: 16,
                overflow: 'hidden',
                transition: 'all 0.25s ease',
                transform: hoveredId === project.id ? 'translateY(-4px)' : 'none',
                boxShadow: hoveredId === project.id ? `0 16px 40px ${project.accent}22` : 'none',
              }}
            >
              {/* Top accent bar */}
              <div style={{ height: 4, background: project.accent, opacity: 0.85 }} />

              <div style={{ padding: 28 }}>
                {/* Header */}
                <div style={{ display: 'flex', alignItems: 'flex-start', gap: 16, marginBottom: 16 }}>
                  <div style={{
                    width: 52,
                    height: 52,
                    borderRadius: 12,
                    background: project.accent + '22',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: 26,
                    flexShrink: 0,
                  }}>
                    {project.emoji}
                  </div>
                  <div>
                    <div style={{ fontSize: 11, fontWeight: 600, color: project.accent, textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 4 }}>
                      {project.category}
                    </div>
                    <h2 style={{ fontSize: 17, fontWeight: 700, color: 'var(--text)', lineHeight: 1.3 }}>
                      {project.title}
                    </h2>
                  </div>
                </div>

                {/* Tagline */}
                <p style={{ fontSize: 14, color: 'var(--text-muted)', marginBottom: 20, lineHeight: 1.5 }}>
                  {project.description}
                </p>

                {/* Metrics */}
                <div style={{ display: 'flex', gap: 12, marginBottom: 20, flexWrap: 'wrap' }}>
                  {project.metrics.map(m => (
                    <div key={m.label} style={{
                      background: 'var(--surface2)',
                      border: '1px solid var(--border)',
                      borderRadius: 8,
                      padding: '8px 14px',
                      flex: 1,
                      minWidth: 80,
                    }}>
                      <div style={{ fontSize: 18, fontWeight: 700, color: project.accent }}>{m.value}</div>
                      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>{m.label}</div>
                    </div>
                  ))}
                </div>

                {/* Tech stack */}
                <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 24 }}>
                  {project.tech.map(t => (
                    <span key={t} style={{
                      padding: '3px 10px',
                      background: 'var(--surface2)',
                      border: '1px solid var(--border)',
                      borderRadius: 6,
                      fontSize: 12,
                      color: 'var(--text-muted)',
                      fontWeight: 500,
                    }}>
                      {t}
                    </span>
                  ))}
                </div>

                {/* Actions */}
                <div style={{ display: 'flex', gap: 10 }}>
                  <a
                    href={project.demoUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{
                      flex: 1,
                      padding: '10px',
                      background: project.accent,
                      borderRadius: 8,
                      textAlign: 'center',
                      fontWeight: 600,
                      fontSize: 13,
                      color: '#fff',
                      transition: 'opacity 0.2s',
                    }}
                  >
                    Live Demo
                  </a>
                  <a
                    href={project.githubUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{
                      flex: 1,
                      padding: '10px',
                      background: 'var(--surface2)',
                      border: '1px solid var(--border)',
                      borderRadius: 8,
                      textAlign: 'center',
                      fontWeight: 600,
                      fontSize: 13,
                      color: 'var(--text-muted)',
                      transition: 'all 0.2s',
                    }}
                  >
                    Source Code
                  </a>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Contact / Footer */}
      <footer style={{
        borderTop: '1px solid var(--border)',
        padding: '40px 24px',
        textAlign: 'center',
        background: 'var(--surface)',
      }}>
        <p style={{ color: 'var(--text-muted)', fontSize: 14, marginBottom: 16 }}>
          Built by <strong style={{ color: 'var(--text)' }}>Oluwafemi Adeyemi</strong> — MIT Applied AI & Data Science Program
        </p>
        <div style={{ display: 'flex', gap: 24, justifyContent: 'center' }}>
          <a href="mailto:femi@phoxta.com" style={{ color: 'var(--primary)', fontSize: 14, fontWeight: 500 }}>femi@phoxta.com</a>
          <a href="https://github.com/oluwafemiadeyemi" target="_blank" rel="noopener noreferrer" style={{ color: 'var(--primary)', fontSize: 14, fontWeight: 500 }}>GitHub</a>
          <a href="https://www.linkedin.com/in/oluwafemiadeyemi" target="_blank" rel="noopener noreferrer" style={{ color: 'var(--primary)', fontSize: 14, fontWeight: 500 }}>LinkedIn</a>
        </div>
      </footer>
    </main>
  )
}
