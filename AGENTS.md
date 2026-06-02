# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

This is a QLoRA fine-tuning project for Qwen2.5-7B-Instruct in the Chinese mine safety domain. The project uses Llama-Factory as the training framework and was run on AutoDL cloud GPU. It is a research/academic project (not a software project) — there is no application code, build system, or test suite.

## Repository Structure

- `data/` — Training data: original PDFs, Markdown conversions, and the final Alpaca-format JSON dataset
- `Config and Index/` — CSV files with training configs and metrics for 3 experiments (test1/test2/test3)
- `Evaluation/` — Before/after comparison: 15 test questions, model answers, GPT-5.5 blind evaluation
- `Export/` — Merged model weights as `.tar` archives (~14 GB each)
- `figure/` — Training loss curve plots (PNG)

## Key Files

- `data/JSON/mine_safety_data.json` — The training dataset (~7265 QA pairs, Alpaca format with `<think>` chains)
- `data/JSON/dataset_info.json` — Llama-Factory dataset registry
- `Config and Index/test*-config.csv` — Full training config for each experiment
- `Config and Index/test*-index.csv` — Training metrics (loss, grad_norm, runtime)
- `Evaluation/Evaluation Result.md` — Complete GPT-5.5 blind evaluation report

## No Build/Test System

This repo has no code to build, lint, or test. The training was done via Llama-Factory's Gradio webui on a remote GPU server. The CSV configs and index files are the primary artifacts.
