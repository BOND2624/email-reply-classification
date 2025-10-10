# Customer Call Transcript Analyzer

AI-powered system that analyzes customer service call transcripts to generate summaries and detect sentiment using Groq API.

## Features

- **AI Analysis**: Summarizes conversations and detects customer sentiment (positive/negative/neutral)
- **Dual Interfaces**: Streamlit (modern UI) or Flask (traditional web app)
- **Data Storage**: Saves results to CSV with timestamps
- **Sample Data**: Includes 6 example transcripts for testing

## Quick Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Get Groq API key**
   - Sign up at [console.groq.com](https://console.groq.com/)
   - Add your key to `.env` file:
   ```
   GROQ_API_KEY=your_key_here
   ```

3. **Run the app**
   ```bash
   # Streamlit (recommended)
   streamlit run streamlit_app.py

## Usage

1. Choose to paste your own transcript or use a sample
2. Click "Analyze" to get AI-generated summary and sentiment
3. View history of past analyses
4. Download results as CSV

## Sample Input/Output

**Input:**
```
Agent: Hello, how can I help?
Customer: I'm frustrated! My service has been down for hours!
Agent: I apologize. Let me fix this right away.
```

**Output:**
- **Summary**: "Customer experienced service outage and expressed frustration. Agent apologized and offered immediate assistance."
- **Sentiment**: Negative

## Files

- `streamlit_app.py` - Modern Streamlit interface
- `app.py` - Flask web application with API
- `sample_transcripts.txt` - Example data for testing
- `transcript_analysis.csv` - Generated results



Built with Python, Groq AI and Streamlit.