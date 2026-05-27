# Deployment Guide

This folder contains everything needed to publish the portfolio publicly.

## Architecture

```
Vercel (portfolio-website/)          ← Landing page linking all demos
    ↓ links to 10 Hugging Face Spaces
HF Space: fair-mortgage              ← Project 1 demo
HF Space: fraud-detection            ← Project 2 demo
HF Space: people-analytics           ← Project 3 demo
HF Space: parkinsons-biomarker       ← Project 4 demo
HF Space: supply-chain-risk          ← Project 5 demo
HF Space: clv-retention              ← Project 6 demo
HF Space: brand-intelligence         ← Project 7 demo
HF Space: retail-operations          ← Project 8 demo
HF Space: workplace-ergonomics       ← Project 9 demo
HF Space: ppe-safety                 ← Project 10 demo
```

## Step 1 — Create Hugging Face Account & Spaces

1. Sign up at https://huggingface.co
2. For each project, create a new Space:
   - Go to https://huggingface.co/new-space
   - Owner: your username
   - Space name: (see names in `hf-spaces/` subfolder)
   - SDK: **Streamlit**
   - Hardware: **CPU Basic** (free)
3. Upload files from `hf-spaces/<project-name>/` to the Space

## Step 2 — Deploy Portfolio Website to Vercel

1. Install Vercel CLI: `npm install -g vercel`
2. `cd portfolio-website && npm install && vercel --prod`
3. Update `portfolio-website/src/data/projects.ts` with your actual HF Space URLs

## Step 3 — Custom Domain (optional)

- Buy `oluwafemiadeyemi.com` on Namecheap (~$12/yr)
- Add domain in Vercel dashboard → Settings → Domains
