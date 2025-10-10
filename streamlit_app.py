import streamlit as st
import os
import csv
import pandas as pd
from datetime import datetime
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Basic page config
st.set_page_config(page_title="Transcript Analyzer", page_icon="📞")

class SimpleAnalyzer:
    def __init__(self):
        self.csv_file = 'call_analysis.csv'
        self.client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        self.ensure_csv_exists()
    
    def ensure_csv_exists(self):
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['Timestamp', 'Transcript', 'Summary', 'Sentiment'])
        else:
            # Check if file has headers
            try:
                df = pd.read_csv(self.csv_file)
                if 'Sentiment' not in df.columns:
                    # File exists but no headers, add them
                    self.fix_csv_headers()
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                # File is empty or corrupted, recreate it
                with open(self.csv_file, 'w', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow(['Timestamp', 'Transcript', 'Summary', 'Sentiment'])
    
    def fix_csv_headers(self):
        # Read existing data
        with open(self.csv_file, 'r', encoding='utf-8') as file:
            existing_data = file.read().strip()
        
        # Rewrite with headers
        with open(self.csv_file, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Timestamp', 'Transcript', 'Summary', 'Sentiment'])
            
            # Add existing data back
            if existing_data:
                for line in existing_data.split('\n'):
                    if line.strip():
                        # Parse the existing CSV line and write it back
                        row = list(csv.reader([line]))[0]
                        writer.writerow(row)
    
    def analyze(self, transcript):
        try:
            # Get summary with enhanced handling for partial conversations
            summary_response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system", 
                        "content": """You are a customer service analyst. Summarize the customer interaction in 1-2 concise sentences.

RULES:
- If it's a full conversation between Agent and Customer, summarize the entire interaction
- If it's just a customer statement or partial conversation, summarize what the customer is experiencing or requesting
- Focus on the customer's issue, request, or situation
- Be concise and factual
- Don't mention missing information - work with what's provided

EXAMPLES:
Input: "Hi, I was trying to book a slot yesterday but the payment failed"
Output: "Customer experienced a payment failure while attempting to book a slot."

Input: "Agent: Hello, how can I help? Customer: My internet is down for 3 hours"
Output: "Customer reported internet service outage lasting 3 hours."
"""
                    },
                    {"role": "user", "content": f"Summarize this customer interaction:\n\n{transcript}"}
                ],
                model="llama-3.1-8b-instant",
                temperature=0.3,
                max_tokens=100
            )
            
            # Get sentiment
            sentiment_response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Analyze customer sentiment. Reply with only: positive, negative, or neutral"},
                    {"role": "user", "content": transcript}
                ],
                model="llama-3.1-8b-instant",
                temperature=0.0,
                max_tokens=10
            )
            
            summary = summary_response.choices[0].message.content.strip()
            sentiment = sentiment_response.choices[0].message.content.strip().lower()
            
            if sentiment not in ['positive', 'negative', 'neutral']:
                sentiment = 'neutral'
            
            return summary, sentiment
            
        except Exception as e:
            return f"Error: {str(e)}", "error"
    
    def save(self, transcript, summary, sentiment):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.csv_file, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, transcript, summary, sentiment])

# Initialize
analyzer = SimpleAnalyzer()

# Main App
st.title("📞 Call Transcript Analyzer")

# Navigation
page = st.selectbox("Choose:", ["Analyze New Transcript", "View History"])

if page == "Analyze New Transcript":
    
    # Add clear button at the top
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write("**Step 1:** Choose your input method")
    with col2:
        if st.button("🧹 Clear Form"):
            # Increment counter to reset all form elements
            if 'form_reset_counter' not in st.session_state:
                st.session_state.form_reset_counter = 0
            st.session_state.form_reset_counter += 1
            st.rerun()
    
    # Initialize session state for form reset counter
    if 'form_reset_counter' not in st.session_state:
        st.session_state.form_reset_counter = 0
    
    # Debug info (remove this later)
    if st.session_state.form_reset_counter > 0:
        st.info(f"Form reset #{st.session_state.form_reset_counter} - All fields should be clear")
    
    # Input choice - fix empty label warning
    choice = st.radio("Input method:", ["Type/Paste transcript", "Use example"], 
                     key=f"input_choice_{st.session_state.form_reset_counter}")
    
    transcript = ""
    
    if choice == "Type/Paste transcript":
        st.write("**Step 2:** Paste your transcript below")
        
        # Use empty string as default value and unique key for each reset
        transcript = st.text_area(
            "Transcript:",
            value="",  # Explicitly set empty value
            height=150,
            placeholder="Agent: Hello, how can I help?\nCustomer: I have a problem with...",
            help="Copy and paste the conversation between agent and customer",
            key=f"transcript_input_{st.session_state.form_reset_counter}"
        )
        
    else:
        st.write("**Step 2:** Pick an example")
        examples = {
            "Happy Customer (Full)": "Agent: Hello, how can I help? Customer: Hi, my internet was down but it's working now. Just wanted to say thanks for fixing it so quickly! Agent: You're welcome! Glad we could help.",
            
            "Angry Customer (Full)": "Agent: Hello, how can I help? Customer: I'm really frustrated! My service has been down for 2 days and nobody called me back! Agent: I sincerely apologize. Let me fix this right away. Customer: This is unacceptable!",
            
            "Normal Inquiry (Full)": "Agent: Hello, how can I help? Customer: Hi, I want to know about your pricing plans. Agent: Sure, our basic plan is $29/month. Customer: What's included? Agent: Unlimited calls and 5GB data.",
            
            "Payment Issue (Partial)": "Hi, I was trying to book a slot yesterday but the payment failed and I'm not sure what went wrong.",
            
            "Service Outage (Brief)": "My internet has been down since this morning and I work from home. This is really affecting my productivity.",
            
            "Account Question (Short)": "I got charged twice this month and need help understanding why this happened.",
            
            "Booking Problem (Customer Only)": "I've been trying to reschedule my appointment for 3 days but your website keeps crashing when I submit the form."
        }
        
        selected = st.selectbox("Choose example:", list(examples.keys()), 
                               key=f"example_choice_{st.session_state.form_reset_counter}")
        transcript = examples[selected]
        st.text_area("Preview:", transcript, height=100, disabled=True)
    
    # Analyze button
    st.write("**Step 3:** Analyze the transcript")
    
    if st.button("🔍 Analyze", type="primary"):
        if transcript.strip():
            with st.spinner("Analyzing..."):
                summary, sentiment = analyzer.analyze(transcript)
                
                if sentiment != "error":
                    # Save result
                    analyzer.save(transcript, summary, sentiment)
                    
                    # Show results
                    st.success("✅ Done!")
                    
                    st.write("**Summary:**")
                    st.write(summary)
                    
                    st.write("**Customer Sentiment:**")
                    if sentiment == "positive":
                        st.success(f"😊 {sentiment.title()}")
                    elif sentiment == "negative":
                        st.error(f"😠 {sentiment.title()}")
                    else:
                        st.warning(f"😐 {sentiment.title()}")
                    
                    # Next steps
                    st.write("---")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🔄 Analyze Another"):
                            # Increment counter to create new form with fresh keys
                            st.session_state.form_reset_counter += 1
                            st.rerun()
                    with col2:
                        if st.button("📊 View All Results"):
                            st.session_state.page = "View History"
                            st.rerun()
                else:
                    st.error(summary)
        else:
            st.warning("⚠️ Please enter a transcript first")

else:  # View History
    st.write("**Your Analysis History**")
    
    try:
        df = pd.read_csv(analyzer.csv_file)
        
        if not df.empty:
            # Check if required columns exist
            if 'Sentiment' not in df.columns:
                st.error("CSV file is missing required columns. Please delete call_analysis.csv and restart the app.")
            else:
                # Simple stats
                total = len(df)
                positive = len(df[df['Sentiment'] == 'positive'])
                negative = len(df[df['Sentiment'] == 'negative'])
                neutral = len(df[df['Sentiment'] == 'neutral'])
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total", total)
                col2.metric("😊 Positive", positive)
                col3.metric("😠 Negative", negative)
                col4.metric("�a Neutral", neutral)
                
                # Recent results
                st.write("**Recent Results:**")
                recent = df.tail(10).copy()
                recent['Short Transcript'] = recent['Transcript'].str[:50] + "..."
                
                # Display table
                for i, row in recent.iterrows():
                    with st.expander(f"{row['Timestamp']} - {row['Sentiment'].title()}"):
                        st.write(f"**Summary:** {row['Summary']}")
                        st.write(f"**Sentiment:** {row['Sentiment'].title()}")
                        st.text_area("Full Transcript:", row['Transcript'], height=100, disabled=True, key=f"transcript_{i}")
                
                # Download
                csv_data = df.to_csv(index=False)
                st.download_button(
                    "📥 Download All Data (CSV)",
                    csv_data,
                    f"analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv"
                )
                
                # Back to analyze
                if st.button("➕ Analyze New Transcript"):
                    st.session_state.page = "Analyze New Transcript"
                    st.rerun()
                
        else:
            st.info("No results yet. Analyze your first transcript!")
            if st.button("🚀 Start Analyzing"):
                st.session_state.page = "Analyze New Transcript"
                st.rerun()
                
    except FileNotFoundError:
        st.info("No results yet. Analyze your first transcript!")
        if st.button("🚀 Start Analyzing"):
            st.session_state.page = "Analyze New Transcript"
            st.rerun()

# Footer
st.write("---")
st.caption("Powered by Groq AI • Built with Streamlit")