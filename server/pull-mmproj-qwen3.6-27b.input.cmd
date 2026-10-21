@echo off
rem pull-mmproj-qwen3.6-27b.input.cmd - 2026-10-16 (Carson: can the 3.6 27B take a picture?): the vision projector for
rem Qwen3.6-27B from the same Unsloth repo as the UD-Q4_K_XL already on C: (928 MB), into the model's folder; the
rem models.input.ini entry gets `mmproj = qwen3.6-27b\mmproj-F16.gguf` like qwen3.5-9b. Resumable.
set DEST=C:\Users\claude\models-b60\qwen3.6-27b
"C:\Users\claude\scripts\vllm-xpu-windows\.venv312\Scripts\python.exe" -c "from huggingface_hub import hf_hub_download; p=hf_hub_download('unsloth/Qwen3.6-27B-GGUF','mmproj-F16.gguf',local_dir=r'%DEST%'); print('done', p)"
echo exit %ERRORLEVEL%
