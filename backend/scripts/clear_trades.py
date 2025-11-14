"""Utility script to purge all trade and fill records from the SQLite database.

Usage (from repo root):
	python backend/scripts/clear_trades.py

Safety:
	- This irreversibly deletes ALL rows in trades and trade_fills tables.
	- Make a backup copy of backend/data/data.db first if needed.
"""

from sqlmodel import create_engine
from sqlalchemy import delete
import sys
from pathlib import Path

# Ensure repo root on sys.path so 'backend' package can be imported when executed from various CWDs
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]
if str(_ROOT) not in sys.path:
	sys.path.insert(0, str(_ROOT))

from backend.app.models import Trade, TradeFill
from backend.app.core.config import settings

def main():
		engine = create_engine(settings.DB_URL)
		# Use a connection transaction for generic execution without Session typing constraints
		with engine.begin() as conn:
			# Delete fills first due to FK constraint
			conn.execute(delete(TradeFill))
			conn.execute(delete(Trade))
		print("All trades and fills deleted.")

if __name__ == "__main__":
		main()

