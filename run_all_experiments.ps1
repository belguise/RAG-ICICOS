# ==============================================================================
# Master Experiment Runner for Q1 Journal Hardening (Windows PowerShell)
# Paper: Explainable Enterprise Process Intelligence via Self-Routing Agentic RAG
# ==============================================================================

Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "STARTING FULL EXPERIMENTAL REPRODUCIBILITY PIPELINE" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan

Write-Host "[1/5] Running Statistical Analysis & Metric Standardization..." -ForegroundColor Yellow
python -m evaluation.statistical_analysis

Write-Host "[2/5] Running Router Sensitivity Analysis (Configs A, B, C, D)..." -ForegroundColor Yellow
python evaluation/experiments/router_sensitivity.py

Write-Host "[3/5] Running Computational Efficiency & Latency Profiling..." -ForegroundColor Yellow
python evaluation/experiments/efficiency_analysis.py

Write-Host "[4/5] Generating Qualitative Error Analysis & BPMN Case Studies..." -ForegroundColor Yellow
python evaluation/generate_error_analysis.py
python evaluation/generate_bpmn_case_studies.py
python evaluation/generate_ablation_analysis.py

Write-Host "[5/5] Generating Publication-Quality Figures (300 DPI PNG & SVG)..." -ForegroundColor Yellow
python evaluation/visualization/generate_q1_paper_figures.py

Write-Host "========================================================================" -ForegroundColor Green
Write-Host "ALL EXPERIMENTS COMPLETED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "Outputs are available in: results/ and paper_figures/" -ForegroundColor Green
Write-Host "========================================================================" -ForegroundColor Green
