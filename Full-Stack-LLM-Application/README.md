---
title: MedIntel
emoji: 🐨
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 6.12.0
app_file: app.py
pinned: false
short_description: MedIntel Q&A Assistant
preload_from_hub:
  - openai/whisper-large-v3-turbo
  - Salesforce/blip2-opt-2.7b
  - sentence-transformers/all-MiniLM-L6-v2
  
app_port: 7860  # This is what actually matters
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
