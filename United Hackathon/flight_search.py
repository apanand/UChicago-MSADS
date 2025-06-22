from flask import Flask, request, jsonify
import os
import google.generativeai as genai
import requests
import datetime
import json

# --- Configuration ---
# To run this, you'll need API keys for Google AI (Gemini) and Amadeus.
# 1. Get a Google AI API key: https://makersuite.google.com/app/apikey
# 2. Get Amadeus API keys: https://developers.amadeus.com/get-started/get-started-with-amadeus-apis-3151
# Set these as environment variables before running the script.
# For example, in your terminal:
# export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
# export AMADEUS_API_KEY="YOUR_AMADEUS_API_KEY"
# export AMADEUS_API_SECRET="YOUR_AMADEUS_API_SECRET"

app = Flask(__name__)

# --- Amadeus API Setup ---
AMADEUS_API_KEY = os.environ.get("AMADEUS_API_KEY")
AMADEUS_API_SECRET = os.environ.get("AMADEUS_API_SECRET")
AMADEUS_TOKEN_URL = "https://test.api.amadeus.com/v1/security/oauth2/token"
AMADEUS_FLIGHT_OFFERS_URL = "https://test.api.amadeus.com/v2/shopping/flight-offers"

def get_amadeus_token():
    """
    Retrieves an OAuth2 token from the Amadeus API.
    """
    payload = {
        'grant_type': 'client_credentials',
        'client_id': AMADEUS_API_KEY,
        'client_secret': AMADEUS_API_SECRET
    }
    try:
        response = requests.post(AMADEUS_TOKEN_URL, data=payload)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json().get('access_token')
    except requests.exceptions.RequestException as e:
        print(f"Error getting Amadeus token: {e}")
        return None

# --- Gemini LLM Setup ---
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

def get_flight_info_from_gemini(user_input):
    """
    Uses Gemini to extract structured flight search parameters from natural language.
    """
    model = genai.GenerativeModel('gemini-pro')

    prompt = f"""
    Extract the following information from the user's request:
    - Origin city IATA code (e.g., DFW for Dallas/Fort Worth).
    - Destination city IATA code (e.g., LAX for Los Angeles).
    - Departure date in YYYY-MM-DD format.
    - Number of adults.

    If any information is missing, use a reasonable default. For example, if the date is not specified, assume tomorrow. If the number of adults is not specified, assume 1.

    User request: "{user_input}"

    Return the information in a JSON format.
    """
    try:
        response = model.generate_content(prompt)
        # Extract the JSON part from the response
        json_response = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(json_response)
    except Exception as e:
        print(f"Error processing with Gemini: {e}")
        return None

# --- Flight Search ---
def search_flights(origin, destination, departure_date, adults):
    """
    Searches for flights using the Amadeus API.
    """
    token = get_amadeus_token()
    if not token:
        return {"error": "Could not authenticate with Amadeus."}

    headers = {
        "Authorization": f"Bearer {token}"
    }

    params = {
        "originLocationCode": origin,
        "destinationLocationCode": destination,
        "departureDate": departure_date,
        "adults": adults,
        "currencyCode": "USD", # Specify the currency as USD
        "max": 5 # Limit the number of results
    }

    try:
        response = requests.get(AMADEUS_FLIGHT_OFFERS_URL, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error searching for flights: {e}")
        return {"error": f"Error from Amadeus API: {e.response.text if e.response else 'No response'}"}


# --- Flask Routes ---
@app.route('/')
def index():
    """Serves the main HTML page."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Flight Search AI</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gray-100 flex items-center justify-center min-h-screen">
        <div class="container mx-auto p-4">
            <div class="bg-white rounded-lg shadow-lg p-8 max-w-2xl mx-auto">
                <h1 class="text-3xl font-bold mb-6 text-center text-gray-800">Flight Search AI</h1>
                <div class="mb-4">
                    <input type="text" id="userInput" class="w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500" placeholder="e.g., 'Fly from DFW to LAX tomorrow for 2 people'">
                </div>
                <div class="text-center">
                    <button id="searchButton" class="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50">Search Flights</button>
                </div>
                <div id="results" class="mt-8"></div>
                 <div id="loading" class="mt-4 text-center" style="display: none;">
                    <p class="text-gray-600">Searching for flights...</p>
                </div>
            </div>
        </div>
        <script>
            document.getElementById('searchButton').addEventListener('click', async () => {
                const userInput = document.getElementById('userInput').value;
                const resultsDiv = document.getElementById('results');
                const loadingDiv = document.getElementById('loading');

                resultsDiv.innerHTML = '';
                loadingDiv.style.display = 'block';

                try {
                    const response = await fetch('/search', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ query: userInput })
                    });
                    const data = await response.json();
                    displayResults(data);
                } catch (error) {
                    resultsDiv.innerHTML = `<p class="text-red-500">An error occurred. See console for details.</p>`;
                    console.error("Error:", error);
                } finally {
                    loadingDiv.style.display = 'none';
                }
            });

            function displayResults(data) {
                const resultsDiv = document.getElementById('results');
                if (data.error) {
                    resultsDiv.innerHTML = `<p class="text-red-500">Error: ${data.error}</p>`;
                    return;
                }

                if (!data.data || data.data.length === 0) {
                    resultsDiv.innerHTML = '<p class="text-gray-600">No flights found.</p>';
                    return;
                }

                let html = '<h2 class="text-2xl font-semibold mb-4 text-gray-700">Flight Options</h2>';
                data.data.forEach(flight => {
                    const itinerary = flight.itineraries[0];
                    const segments = itinerary.segments;
                    const airlineCodes = [...new Set(segments.map(s => s.carrierCode))].join(', ');

                    html += `
                        <div class="border rounded-lg p-4 mb-4 bg-gray-50">
                            <p><strong>Price:</strong> ${flight.price.total} ${flight.price.currency}</p>
                            <p><strong>Airlines:</strong> ${airlineCodes}</p>
                            <p><strong>Stops:</strong> ${segments.length - 1}</p>
                         </div>`;
                });
                resultsDiv.innerHTML = html;
            }
        </script>
    </body>
    </html>
    """

@app.route('/search', methods=['POST'])
def handle_search():
    """
    Handles the flight search request from the frontend.
    """
    user_query = request.json.get('query')
    if not user_query:
        return jsonify({"error": "No query provided."}), 400

    # Get structured data from Gemini
    flight_params = get_flight_info_from_gemini(user_query)
    if not flight_params:
         # Fallback with default values if Gemini fails
        flight_params = {
            "origin": "DFW",
            "destination": "LAX",
            "departure_date": (datetime.date.today() + datetime.timedelta(days=1)).strftime('%Y-%m-%d'),
            "adults": 1
        }
    
    # Search for flights using Amadeus
    flight_data = search_flights(
        flight_params.get('origin'),
        flight_params.get('destination'),
        flight_params.get('departure_date'),
        flight_params.get('adults')
    )

    return jsonify(flight_data)

if __name__ == '__main__':
    # Make sure to set the environment variables before running!
    if not all([os.environ.get("GEMINI_API_KEY"), os.environ.get("AMADEUS_API_KEY"), os.environ.get("AMADEUS_API_SECRET")]):
        print("ERROR: Make sure you have set GEMINI_API_KEY, AMADEUS_API_KEY, and AMADEUS_API_SECRET environment variables.")
    else:
        app.run(debug=True)

def run(input_data: dict) -> dict:
    """
    Fast MCP-compatible run function for flight search.
    Requires: destination, start_date, end_date
    """
    from datetime import datetime
    origin = input_data.get("origin", "ORD")  # default to Chicago O'Hare
    destination = input_data.get("destination", "LAX")
    departure_date = input_data.get("start_date", str(datetime.today().date()))
    return_date = input_data.get("end_date", str(datetime.today().date()))

    token = get_amadeus_token()
    if not token:
        return {"error": "Failed to authenticate with Amadeus"}

    params = {
        "originLocationCode": origin,
        "destinationLocationCode": destination,
        "departureDate": departure_date,
        "returnDate": return_date,
        "adults": 1,
        "max": 3
    }

    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(AMADEUS_FLIGHT_OFFERS_URL, headers=headers, params=params)

    try:
        data = response.json()
        return {"flights": data.get("data", [])}
    except Exception as e:
        return {"error": str(e)}
