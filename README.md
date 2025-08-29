# FRP Streamlit Starter (with DB & Deploy scaffold)

## Files
- `app.py` — your Streamlit app (copied from the uploaded file; replace if needed)
- `requirements.txt` — Python dependencies
- `.env.example` — copy to `.env` and fill DB credentials
- `.gitignore` — keeps `.env` out of Git
- `research_data_sample.csv` — a tiny sample dataset you can import to test DB connectivity
- `create_database.sql` — creates the MySQL database `haigui_database`

## Quickstart
1. Create venv and install deps:
```
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```
2. Copy `.env.example` to `.env` and fill DB settings.
3. Ensure MySQL is running; create DB with `create_database.sql`.
4. Use MySQL Workbench 'Table Data Import Wizard' to import `research_data_sample.csv` as table `research_data` in `haigui_database`.
5. Run locally:
```
streamlit run app.py
```
6. Push this folder to GitHub, then deploy on Streamlit Cloud and set the same environment variables there.
