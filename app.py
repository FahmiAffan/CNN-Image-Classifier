from app import create_app
import os

# Create the Flask app
app = create_app()

# Configure for Hugging Face Spaces
if __name__ == "__main__":
    # For Hugging Face Spaces deployment
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
