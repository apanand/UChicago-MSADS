"""
Pre-Trip Requirements Agent - Streamlit POC with Groq Llama
United Airlines Hackathon Entry

AI-powered travel requirements research using Groq's Llama model
for dynamic, real-time travel requirements analysis.
"""

import streamlit as st
import asyncio
import json
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import os
from dotenv import load_dotenv
import logging
import requests
from groq import Groq

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Groq API Configuration
GROQ_API_KEY = "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"

# ============================================================================
# Data Models
# ============================================================================

@dataclass
class TravelRequirement:
    requirement_type: str
    status: str  # required, recommended, not_required
    description: str
    details: str = ""
    processing_time: str = ""
    cost: str = ""
    authority: str = ""
    conditions: List[str] = None
    
    def __post_init__(self):
        if self.conditions is None:
            self.conditions = []

@dataclass
class PassengerProfile:
    nationality: str
    destination: str
    departure_date: date
    trip_purpose: str = "Tourism"
    passport_expiry: Optional[date] = None

# ============================================================================
# Groq-Powered Travel Requirements Client
# ============================================================================

class GroqTravelRequirementsClient:
    """AI-powered travel requirements client using Groq Llama model"""
    
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.model = "llama3-70b-8192"  # Using Llama 3.1 70B
        
        # Country code mappings for better LLM understanding
        self.country_names = {
            "US": "United States",
            "IN": "India", 
            "GB": "United Kingdom",
            "CN": "China",
            "FR": "France",
            "DE": "Germany",
            "JP": "Japan",
            "AU": "Australia",
            "BR": "Brazil",
            "CA": "Canada"
        }
    
    async def get_travel_requirements(self, nationality: str, destination: str, departure_date: date, trip_purpose: str = "Tourism") -> List[TravelRequirement]:
        """Get AI-powered travel requirements analysis"""
        
        try:
            # Convert country codes to full names for better LLM understanding
            nationality_name = self.country_names.get(nationality, nationality)
            destination_name = self.country_names.get(destination, destination)
            
            # Calculate days until travel for context
            days_until_travel = (departure_date - date.today()).days
            
            # Create comprehensive prompt for the LLM
            prompt = self._create_requirements_prompt(
                nationality_name, destination_name, departure_date, trip_purpose, days_until_travel
            )
            
            # Call Groq API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Low temperature for factual accuracy
                max_tokens=2000,
                top_p=0.9
            )
            
            # Parse the AI response
            ai_response = response.choices[0].message.content
            logger.info(f"Groq AI Response received: {len(ai_response)} characters")
            
            # Convert AI response to structured requirements
            requirements = self._parse_ai_response(ai_response)
            
            return requirements
            
        except Exception as e:
            logger.error(f"Error getting AI requirements: {e}")
            # Fallback to basic requirements if AI fails
            return self._get_fallback_requirements(nationality, destination)
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the AI agent"""
        
        return """You are an expert travel requirements research agent working for United Airlines. Your role is to provide accurate, up-to-date travel requirements for international passengers.

CRITICAL INSTRUCTIONS:
1. Provide factual, current travel requirements based on official government sources
2. Focus on the most important requirements: visa, passport, health/vaccination, customs
3. Include specific processing times, costs, and authorities where known
4. Highlight urgent requirements based on departure timeline
5. Be conservative - if unsure, recommend passengers verify with official sources

FORMAT YOUR RESPONSE AS JSON with this exact structure:
{
  "requirements": [
    {
      "requirement_type": "visa|passport|vaccination|health_certificate|covid_test|customs",
      "status": "required|recommended|not_required|conditional",
      "description": "Brief description",
      "details": "Detailed explanation", 
      "processing_time": "Time needed",
      "cost": "Cost range",
      "authority": "Issuing authority",
      "conditions": ["list", "of", "conditions"]
    }
  ],
  "urgent_items": ["list of urgent requirements"],
  "timeline_warnings": ["warnings based on departure date"],
  "additional_notes": "Any important additional information"
}

Remember: Accuracy is critical for international travel. When in doubt, advise passengers to verify with official embassy/consulate sources."""

    def _create_requirements_prompt(self, nationality: str, destination: str, departure_date: date, trip_purpose: str, days_until_travel: int) -> str:
        """Create detailed prompt for travel requirements research"""
        
        return f"""Analyze travel requirements for the following trip:

PASSENGER DETAILS:
- Nationality: {nationality} citizen
- Destination: {destination}
- Departure Date: {departure_date.strftime('%B %d, %Y')} ({days_until_travel} days from today)
- Trip Purpose: {trip_purpose}
- Current Date: {date.today().strftime('%B %d, %Y')}

RESEARCH REQUIREMENTS FOR:
1. VISA REQUIREMENTS
   - Is a visa required for {nationality} citizens visiting {destination}?
   - What type of visa (tourist, business, transit)?
   - Processing time and cost?
   - Required documents?

2. PASSPORT REQUIREMENTS  
   - Validity period required (6 months rule?)
   - Blank pages needed?
   - Special conditions?

3. HEALTH REQUIREMENTS
   - Required vaccinations?
   - COVID-19 requirements?
   - Health certificates needed?

4. ENTRY REQUIREMENTS
   - Proof of onward travel?
   - Minimum funds requirement?
   - Travel insurance mandatory?

5. TIMELINE ANALYSIS
   - With {days_until_travel} days until departure, what's urgent?
   - What can still be completed in time?
   - What might require expedited processing?

Please provide comprehensive, accurate requirements based on current regulations. If any requirement has changed recently or varies by entry point, please note this."""

    def _parse_ai_response(self, ai_response: str) -> List[TravelRequirement]:
        """Parse AI response into structured requirements"""
        
        try:
            # Try to extract JSON from the response
            json_start = ai_response.find('{')
            json_end = ai_response.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = ai_response[json_start:json_end]
                parsed_data = json.loads(json_str)
                
                requirements = []
                for req_data in parsed_data.get('requirements', []):
                    requirement = TravelRequirement(
                        requirement_type=req_data.get('requirement_type', 'other'),
                        status=req_data.get('status', 'unknown'),
                        description=req_data.get('description', ''),
                        details=req_data.get('details', ''),
                        processing_time=req_data.get('processing_time', ''),
                        cost=req_data.get('cost', ''),
                        authority=req_data.get('authority', ''),
                        conditions=req_data.get('conditions', [])
                    )
                    requirements.append(requirement)
                
                return requirements
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from AI response: {e}")
        
        # Fallback: Parse text response using pattern matching
        return self._parse_text_response(ai_response)
    
    def _parse_text_response(self, response: str) -> List[TravelRequirement]:
        """Fallback parser for non-JSON AI responses"""
        
        requirements = []
        
        # Look for visa requirements
        if "visa required" in response.lower() or "visa is required" in response.lower():
            requirements.append(TravelRequirement(
                requirement_type="visa",
                status="required",
                description="Visa Required",
                details="Based on AI analysis, a visa is required for this destination",
                processing_time="Check with embassy",
                authority="Embassy/Consulate"
            ))
        elif "no visa" in response.lower() or "visa not required" in response.lower():
            requirements.append(TravelRequirement(
                requirement_type="visa", 
                status="not_required",
                description="No Visa Required",
                details="Based on AI analysis, no visa is required for this destination"
            ))
        
        # Look for passport requirements
        if "passport" in response.lower():
            requirements.append(TravelRequirement(
                requirement_type="passport",
                status="required",
                description="Valid Passport Required",
                details="Passport must be valid for travel",
                authority="Passport issuing authority"
            ))
        
        # Look for vaccination requirements
        if "vaccination" in response.lower() or "vaccine" in response.lower():
            requirements.append(TravelRequirement(
                requirement_type="vaccination",
                status="recommended",
                description="Vaccination Requirements", 
                details="Check vaccination requirements with healthcare provider",
                authority="Healthcare provider/CDC"
            ))
        
        return requirements
    
    def _get_fallback_requirements(self, nationality: str, destination: str) -> List[TravelRequirement]:
        """Fallback requirements if AI fails"""
        
        return [
            TravelRequirement(
                requirement_type="passport",
                status="required",
                description="Valid Passport Required",
                details="Passport must be valid for international travel",
                authority="Passport issuing authority"
            ),
            TravelRequirement(
                requirement_type="visa",
                status="unknown",
                description="Visa Requirements Unknown",
                details="Please check with destination embassy for current visa requirements",
                authority="Destination country embassy"
            )
        ]

# ============================================================================
# AI-Powered Travel Requirements Agent
# ============================================================================

class AITravelRequirementsAgent:
    """AI-powered travel requirements research agent using Groq"""
    
    def __init__(self, groq_api_key: str):
        self.requirements_client = GroqTravelRequirementsClient(groq_api_key)
        self.agent_name = "AI Travel Requirements Researcher"
        self.agent_role = "AI-powered expert in international travel regulations"
    
    async def research_requirements(self, passenger_profile: PassengerProfile) -> Dict[str, Any]:
        """Research travel requirements using AI"""
        
        logger.info(f"AI researching requirements: {passenger_profile.nationality} → {passenger_profile.destination}")
        
        try:
            # Get AI-powered requirements
            requirements = await self.requirements_client.get_travel_requirements(
                passenger_profile.nationality,
                passenger_profile.destination,
                passenger_profile.departure_date,
                passenger_profile.trip_purpose
            )
            
            # Analyze timeline and urgency
            days_until_travel = (passenger_profile.departure_date - date.today()).days
            
            # Generate AI-enhanced recommendations
            recommendations = self._generate_ai_recommendations(requirements, days_until_travel, passenger_profile)
            
            # Assess overall compliance
            compliance_status = self._assess_compliance(requirements, passenger_profile)
            
            return {
                "passenger_profile": {
                    "nationality": passenger_profile.nationality,
                    "destination": passenger_profile.destination,
                    "departure_date": passenger_profile.departure_date.isoformat(),
                    "trip_purpose": passenger_profile.trip_purpose
                },
                "requirements": [
                    {
                        "type": req.requirement_type,
                        "status": req.status,
                        "description": req.description,
                        "details": req.details,
                        "processing_time": req.processing_time,
                        "cost": req.cost,
                        "authority": req.authority,
                        "conditions": req.conditions
                    }
                    for req in requirements
                ],
                "timeline_analysis": {
                    "days_until_travel": days_until_travel,
                    "urgency_level": "high" if days_until_travel < 14 else "medium" if days_until_travel < 45 else "low",
                    "sufficient_time": days_until_travel >= 14
                },
                "recommendations": recommendations,
                "compliance_status": compliance_status,
                "research_timestamp": datetime.now().isoformat(),
                "agent_info": {
                    "name": self.agent_name,
                    "role": self.agent_role,
                    "ai_model": "Groq Llama 3.1 70B",
                    "capabilities": ["Real-time requirements research", "AI-powered analysis", "Dynamic recommendations"]
                }
            }
            
        except Exception as e:
            logger.error(f"Error in AI requirements research: {e}")
            return {
                "error": str(e),
                "status": "failed",
                "timestamp": datetime.now().isoformat(),
                "fallback_message": "AI analysis failed. Please verify requirements with official sources."
            }
    
    def _generate_ai_recommendations(self, requirements: List[TravelRequirement], days_until_travel: int, profile: PassengerProfile) -> List[str]:
        """Generate AI-enhanced recommendations"""
        
        recommendations = []
        
        # Check for visa requirements with timeline
        visa_required = any(req.requirement_type == "visa" and req.status == "required" for req in requirements)
        
        if visa_required:
            if days_until_travel < 7:
                recommendations.append("🚨 CRITICAL: Visa required but only {} days until travel - contact embassy immediately for emergency processing".format(days_until_travel))
            elif days_until_travel < 14:
                recommendations.append("⚠️ URGENT: Apply for visa immediately - consider expedited processing")
            elif days_until_travel < 30:
                recommendations.append("🔔 Apply for visa soon to ensure processing time")
            else:
                recommendations.append("📅 Plan to apply for visa within the next 2 weeks")
        
        # Passport validity recommendations
        passport_req = next((req for req in requirements if req.requirement_type == "passport"), None)
        if passport_req and "6 months" in passport_req.details.lower():
            recommendations.append("📘 Verify passport validity - must be valid 6+ months beyond travel date")
        
        # Health and vaccination recommendations
        health_reqs = [req for req in requirements if req.requirement_type in ["vaccination", "covid_test", "health_certificate"]]
        if health_reqs:
            recommendations.append("💉 Schedule healthcare consultation for required/recommended vaccinations")
        
        # Timeline-specific recommendations
        if days_until_travel < 30:
            recommendations.append("📋 Gather all required documents immediately")
            recommendations.append("📞 Consider contacting destination embassy to confirm current requirements")
        
        # General travel recommendations
        recommendations.extend([
            "📱 Download destination country's official travel app if available",
            "🛡️ Consider comprehensive travel insurance",
            "💳 Notify banks of international travel plans",
            "📧 Register with embassy if traveling to high-risk areas"
        ])
        
        return recommendations[:8]  # Limit to most important recommendations
    
    def _assess_compliance(self, requirements: List[TravelRequirement], profile: PassengerProfile) -> str:
        """Assess overall compliance status using AI analysis"""
        
        required_items = [req for req in requirements if req.status == "required"]
        
        if not required_items:
            return "likely_compliant"
        
        # Check for high-impact requirements
        visa_required = any(req.requirement_type == "visa" for req in required_items)
        days_until_travel = (profile.departure_date - date.today()).days
        
        if visa_required and days_until_travel < 14:
            return "high_risk"
        elif visa_required:
            return "action_required"
        elif days_until_travel < 7:
            return "time_critical"
        
        return "likely_compliant"

# ============================================================================
# Streamlit Application
# ============================================================================

def main():
    """Main Streamlit application"""
    
    st.set_page_config(
        page_title="AI Pre-Trip Requirements Agent",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("🤖 AI Pre-Trip Requirements Agent")
    st.markdown("**United Airlines Hackathon POC** - Powered by Groq Llama 3.1 70B")
    st.markdown("Real-time AI analysis of international travel requirements")
    st.markdown("---")
    
    # Sidebar - Agent Information
    with st.sidebar:
        st.header("🧠 AI Agent Information")
        st.markdown("""
        **Agent:** AI Travel Requirements Researcher  
        **AI Model:** Groq Llama 3.1 70B
        
        **AI Capabilities:**
        - Real-time requirements research
        - Dynamic visa analysis
        - Timeline-based urgency assessment
        - Personalized recommendations
        - Current policy awareness
        
        **Technology Stack:**
        - Groq API (Ultra-fast inference)
        - Llama 3.1 70B model
        - Streamlit interface
        - Async processing
        """)
        
        st.markdown("---")
        st.markdown("**🔥 Live AI Analysis:**")
        st.markdown("• Real visa requirements")
        st.markdown("• Current health policies")
        st.markdown("• Dynamic recommendations")
        st.markdown("• Timeline risk assessment")
        
        # AI Status
        if st.button("🧪 Test AI Connection"):
            with st.spinner("Testing Groq API..."):
                try:
                    client = Groq(api_key=GROQ_API_KEY)
                    response = client.chat.completions.create(
                        model="llama3-70b-8192",
                        messages=[{"role": "user", "content": "Say 'AI connection successful'"}],
                        max_tokens=50
                    )
                    st.success("✅ AI Model Ready!")
                    st.write(f"Response: {response.choices[0].message.content}")
                except Exception as e:
                    st.error(f"❌ AI Connection Failed: {str(e)}")
    
    # Main Content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("✈️ Travel Information")
        
        # Travel input form
        with st.form("travel_form"):
            nationality = st.selectbox(
                "Passenger Nationality",
                ["US", "CA", "GB", "AU", "DE", "FR"],
                index=0,
                help="Passenger's nationality/passport country"
            )
            
            destination = st.selectbox(
                "Destination Country", 
                ["IN", "CN", "GB", "FR", "DE", "JP", "AU", "BR", "RU", "MX"],
                index=0,
                help="Destination country"
            )
            
            departure_date = st.date_input(
                "Departure Date",
                value=date.today() + timedelta(days=30),
                min_value=date.today(),
                max_value=date.today() + timedelta(days=365),
                help="Planned departure date"
            )
            
            trip_purpose = st.selectbox(
                "Trip Purpose",
                ["Tourism", "Business", "Transit", "Study", "Work", "Family Visit"],
                index=0
            )
            
            submit_button = st.form_submit_button("🚀 AI Requirements Analysis", type="primary")
    
    with col2:
        st.header("🎯 AI Analysis Results")
        
        if submit_button:
            # Create passenger profile
            passenger_profile = PassengerProfile(
                nationality=nationality,
                destination=destination,
                departure_date=departure_date,
                trip_purpose=trip_purpose
            )
            
            # Show AI processing status
            with st.spinner("🤖 AI analyzing travel requirements with Groq Llama..."):
                try:
                    # Initialize AI agent
                    agent = AITravelRequirementsAgent(GROQ_API_KEY)
                    
                    # Run AI analysis
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(agent.research_requirements(passenger_profile))
                    loop.close()
                    
                except Exception as e:
                    st.error(f"❌ AI Analysis Failed: {str(e)}")
                    st.stop()
            
            # Display AI results
            if "error" not in result:
                st.success("✅ AI analysis completed!")
                
                # AI Model Info
                agent_info = result.get("agent_info", {})
                st.info(f"🧠 Analysis by: {agent_info.get('ai_model', 'AI Model')}")
                
                # Timeline Analysis
                timeline = result["timeline_analysis"]
                days_until = timeline["days_until_travel"]
                urgency = timeline["urgency_level"]
                
                urgency_colors = {
                    "high": "🔴",
                    "medium": "🟡", 
                    "low": "🟢"
                }
                
                st.metric(
                    "Days Until Travel",
                    f"{days_until} days",
                    delta=f"Urgency: {urgency_colors.get(urgency, '⚪')} {urgency.title()}"
                )
                
                # AI Compliance Assessment
                compliance = result["compliance_status"]
                compliance_display = {
                    "likely_compliant": "✅ Likely Compliant",
                    "action_required": "⚠️ Action Required",
                    "high_risk": "🚨 High Risk",
                    "time_critical": "⏰ Time Critical"
                }
                
                st.metric("AI Assessment", compliance_display.get(compliance, compliance))
                
            else:
                st.error(f"❌ AI Error: {result.get('error', 'Unknown error')}")
                if result.get('fallback_message'):
                    st.warning(result['fallback_message'])
                st.stop()
    
    # AI Results Section
    if submit_button and "error" not in result:
        st.markdown("---")
        st.header("🧠 AI-Generated Requirements Analysis")
        
        # Requirements from AI
        requirements = result["requirements"]
        
        if not requirements:
            st.warning("⚠️ AI couldn't determine specific requirements. Please verify with official sources.")
        
        for req in requirements:
            with st.expander(f"{req['type'].title()}: {req['description']}", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    status_colors = {
                        "required": "🔴",
                        "recommended": "🟡",
                        "not_required": "🟢",
                        "conditional": "🟠",
                        "unknown": "⚪"
                    }
                    st.write(f"**Status:** {status_colors.get(req['status'], '⚪')} {req['status'].title()}")
                    if req['authority']:
                        st.write(f"**Authority:** {req['authority']}")
                
                with col2:
                    if req['processing_time']:
                        st.write(f"**Processing Time:** {req['processing_time']}")
                    if req['cost']:
                        st.write(f"**Cost:** {req['cost']}")
                
                with col3:
                    if req['conditions']:
                        st.write("**Conditions:**")
                        for condition in req['conditions']:
                            st.write(f"• {condition}")
                
                if req['details']:
                    st.info(f"**AI Analysis:** {req['details']}")
        
        # AI Recommendations
        st.markdown("---")
        st.header("🎯 AI-Generated Recommendations")
        
        recommendations = result["recommendations"]
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"{i}. {rec}")
        
        # AI Action Plan
        st.markdown("---")
        st.header("📋 AI Action Plan")
        
        timeline = result["timeline_analysis"]
        compliance = result["compliance_status"]
        
        if compliance == "high_risk":
            st.error("🚨 HIGH RISK: Immediate action required - very limited time!")
        elif compliance == "time_critical":
            st.error("⏰ TIME CRITICAL: Urgent action needed")
        elif compliance == "action_required":
            st.warning("⚠️ Action required to ensure compliance")
        else:
            st.success("✅ Situation manageable with proper planning")
        
        # Dynamic action items based on AI analysis
        st.markdown("**AI-Prioritized Action Items:**")
        
        action_items = []
        
        # Check AI requirements for specific actions
        visa_required = any(req["type"] == "visa" and req["status"] == "required" for req in requirements)
        health_reqs = any(req["type"] in ["vaccination", "covid_test"] for req in requirements)
        
        if visa_required:
            if timeline["days_until_travel"] < 14:
                action_items.append("🚨 EMERGENCY: Contact embassy immediately for expedited visa processing")
            else:
                action_items.append("📋 Apply for visa within 48 hours")
                action_items.append("📸 Gather required documents and photos")
        
        action_items.extend([
            "📘 Verify passport validity requirements",
            "💉 Consult healthcare provider if vaccinations needed" if health_reqs else "✅ No specific health requirements identified",
            "🛡️ Consider comprehensive travel insurance",
            "📱 Register with embassy for high-risk destinations",
            "🔄 Recheck requirements 1 week before travel"
        ])
        
        for i, item in enumerate(action_items, 1):
            st.markdown(f"{i}. {item}")
        
        # AI Confidence and Sources
        st.markdown("---")
        st.header("🎓 AI Analysis Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**AI Model Used:**")
            st.code("Groq Llama 3.1 70B")
            st.markdown("**Analysis Type:**")
            st.code("Real-time requirements research")
        
        with col2:
            st.markdown("**Verification Needed:**")
            st.warning("⚠️ Always verify AI analysis with official embassy/government sources")
            st.markdown("**Last Updated:**")
            st.code(result["research_timestamp"])
        
        # Raw AI data (for technical review)
        with st.expander("🔧 Raw AI Response (Technical)", expanded=False):
            st.json(result)

# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    main()


# Fast MCP-compatible entry point
def run(input_data: dict) -> dict:
    import asyncio

    nationality = input_data.get("nationality", "US")
    destination = input_data.get("destination", "IN")
    start_date_str = input_data.get("start_date", str(date.today() + timedelta(days=30)))
    trip_purpose = input_data.get("trip_purpose", "Tourism")

    # Convert to datetime.date
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()

    # Build the profile
    profile = PassengerProfile(
        nationality=nationality,
        destination=destination,
        departure_date=start_date,
        trip_purpose=trip_purpose
    )

    # Run async Groq agent
    agent = AITravelRequirementsAgent(GROQ_API_KEY)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(agent.research_requirements(profile))
    loop.close()

    return result
