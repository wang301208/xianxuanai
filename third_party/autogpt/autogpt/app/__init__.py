from pathlib import Path
from dotenv import load_dotenv

# Load the root .env file into environment variables
load_dotenv(Path(__file__).resolve().parents[4] / ".env", verbose=True, override=True)

del load_dotenv
