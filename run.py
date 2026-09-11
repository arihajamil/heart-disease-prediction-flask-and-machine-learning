"""
Heart Disease Prediction - development entry point
========================================
Start the application with:

    python run.py

Then open http://127.0.0.1:5000 in a browser.

For production use a real WSGI server instead, for example:
    waitress-serve --port=8000 "run:app"
"""

import os
from routes import create_app

app = create_app()


if __name__ == "__main__":
    port = int(os.environ.get("HDP_PORT", 5000))
    debug = os.environ.get("HDP_ENV", "development").lower().startswith("dev")

    print("=" * 60)
    print("  Heart Disease Prediction - Cardiovascular Risk Portal")
    print("=" * 60)
    print(f"  Running at : http://127.0.0.1:{port}")
    print(f"  Debug mode : {debug}")
    print("  Press CTRL+C to stop")
    print("=" * 60)

    app.run(host="127.0.0.1", port=port, debug=debug)
