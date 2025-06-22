# orchestrator.py

import streamlit as st
from datetime import date
from pre_trip_reqs import run as run_pre_trip
from flight_search import run as run_flight
from itinerary_plan import run as run_itinerary
from booking_interactive import run as run_booking

# Set up page
st.set_page_config(page_title="🌍 Smart Travel Planner", layout="wide")
st.title("🌍 Smart Travel Planner")
st.markdown("Use AI agents to plan your next international trip.")

# Input form
with st.form("trip_form"):
    destination = st.text_input("Destination", "Chicago")
    nationality = st.text_input("Nationality", "US")
    trip_purpose = st.selectbox("Trip Purpose", ["Tourism", "Business", "Work", "Study"], index=0)
    start_date = st.date_input("Start Date", date.today())
    end_date = st.date_input("End Date", date.today())
    submitted = st.form_submit_button("Generate Trip Plan")

# If form is submitted
if submitted:
    st.info("Running multi-agent trip planner...")

    # Initial input
    input_data = {
        "destination": destination,
        "nationality": nationality,
        "trip_purpose": trip_purpose,
        "start_date": str(start_date),
        "end_date": str(end_date),
    }

    # Run each agent step by step
    try:
        input_data.update(run_pre_trip(input_data))
        input_data.update(run_flight(input_data))
        input_data.update(run_itinerary(input_data))
        input_data.update(run_booking(input_data))

        # Display final plan
        st.success("✅ Your AI Trip Plan is Ready!")
        st.json(input_data)

    except Exception as e:
        st.error(f"❌ An error occurred while running the pipeline: {e}")
