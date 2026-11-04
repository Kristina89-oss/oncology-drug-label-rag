# One-click start (key is read from .env next to this file). Open http://localhost:8501
Set-Location $PSScriptRoot
& .\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false
