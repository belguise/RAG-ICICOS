@echo off
REM ==============================================================================
REM Master Experiment Runner for Q1 Journal Hardening (Windows Command Prompt)
REM Paper: Explainable Enterprise Process Intelligence via Self-Routing Agentic RAG
REM ==============================================================================

echo ========================================================================
echo STARTING FULL EXPERIMENTAL REPRODUCIBILITY PIPELINE
echo ========================================================================

echo [1/5] Running Statistical Analysis & Metric Standardization...
python -m evaluation.statistical_analysis

echo [2/5] Running Router Sensitivity Analysis (Configs A, B, C, D)...
python evaluation\experiments\router_sensitivity.py

echo [3/5] Running Computational Efficiency & Latency Profiling...
python evaluation\experiments\efficiency_analysis.py

echo [4/5] Generating Qualitative Error Analysis & BPMN Case Studies...
python evaluation\generate_error_analysis.py
python evaluation\generate_bpmn_case_studies.py
python evaluation\generate_ablation_analysis.py

echo [5/5] Generating Publication-Quality Figures (300 DPI PNG & SVG)...
python evaluation\visualization\generate_q1_paper_figures.py

echo ========================================================================
echo ALL EXPERIMENTS COMPLETED SUCCESSFULLY!
echo Outputs are available in: results\ and paper_figures\
echo ========================================================================
