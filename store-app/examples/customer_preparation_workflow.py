#!/usr/bin/env python3
"""
Customer Preparation Workflow Example

This example demonstrates how store managers can use the retail AI agent
to quickly prepare for any customer visit, including personal shopping
appointments and important customer interactions.

The workflow shows:
1. Finding upcoming customer appointments
2. Getting detailed customer information
3. Creating AI-powered preparation summaries for staff
4. Understanding customer preferences and requirements
"""

import yaml
from mlflow.models import ModelConfig
from retail_ai.tools.factory import ToolFactory
from langchain_databricks import ChatDatabricks

def demonstrate_customer_preparation_workflow():
    """
    Demonstrate the complete customer preparation workflow for store managers.
    """
    print("🏪 Customer Preparation Workflow Demo")
    print("=" * 60)
    print("Scenario: Store manager preparing for upcoming customer appointments")
    print()
    
    # Load configuration
    with open('model_config.yaml', 'r') as f:
        config_dict = yaml.safe_load(f)
    
    config = ModelConfig(development_config=config_dict)
    factory = ToolFactory(config)
    
    # Get warehouse ID from config
    warehouse_id = config.get("resources").get("warehouses").get("shared_endpoint_warehouse").get("warehouse_id")
    
    # Create LLM for AI-powered summaries
    llm_config = config.get("resources").get("llms").get("default_llm")
    llm = ChatDatabricks(
        endpoint=llm_config.get("name"),
        temperature=llm_config.get("temperature", 0.1),
        max_tokens=llm_config.get("max_tokens", 8192)
    )
    
    # Create customer tools with LLM
    customer_tools = factory.create_customer_tools(llm, warehouse_id)
    
    print("📋 Step 1: Check upcoming appointments for today")
    print("-" * 50)
    
    # Find upcoming appointments
    find_appointments = customer_tools["find_upcoming_customer_appointments"]
    
    # Look for appointments in the next 24 hours at store 101
    print("Manager: 'Show me upcoming customer appointments at our downtown store (101) for the next 24 hours'")
    print()
    
    try:
        appointments = find_appointments.invoke({
            "store_id": "101",
            "hours_ahead": 24
        })
        
        if appointments:
            print(f"✅ Found {len(appointments)} upcoming appointments:")
            for i, appointment in enumerate(appointments, 1):
                print(f"\n{i}. {appointment.get('customer_name')} ({appointment.get('customer_tier')} tier)")
                print(f"   📅 {appointment.get('next_appointment_date')}")
                print(f"   🎯 {appointment.get('appointment_type')}: {appointment.get('appointment_purpose')}")
                print(f"   💰 Budget: {appointment.get('budget_range')}")
                if appointment.get('requires_manager_greeting'):
                    print(f"   🤝 Manager greeting required")
                if appointment.get('customer_alerts'):
                    print(f"   ⚠️  Alert: {appointment.get('customer_alerts')}")
        else:
            print("No upcoming appointments found.")
            
    except Exception as e:
        print(f"❌ Error finding appointments: {e}")
        print("Note: This would work with a live database connection.")
    
    print("\n" + "=" * 60)
    print("📊 Step 2: Get detailed customer information")
    print("-" * 50)
    
    # Get detailed customer information
    get_details = customer_tools["get_customer_details"]
    
    print("Manager: 'Tell me more about Sarah Johnson who has an appointment today'")
    print()
    
    try:
        customer_details = get_details.invoke({
            "customer_name": "Sarah Johnson"
        })
        
        if customer_details:
            customer = customer_details[0]
            print(f"✅ Customer Details for {customer.get('customer_name')}:")
            print(f"   👤 Preferred name: {customer.get('preferred_name')}")
            print(f"   🏆 Tier: {customer.get('customer_tier')}")
            print(f"   💳 Lifetime spend: ${customer.get('total_lifetime_spend', 0):,.2f}")
            print(f"   ⭐ Satisfaction: {customer.get('customer_satisfaction', 0):.1f}/5.0")
            print(f"   👗 Preferred stylist: {customer.get('preferred_stylist_name')}")
            print(f"   🎨 Style preferences: {customer.get('style_preferences')}")
            print(f"   💰 Budget range: {customer.get('budget_range')}")
            
            if customer.get('special_occasions'):
                print(f"   🎉 Special occasions: {customer.get('special_occasions')}")
            if customer.get('dietary_restrictions'):
                print(f"   🥗 Dietary restrictions: {customer.get('dietary_restrictions')}")
                
        else:
            print("Customer not found in database.")
            
    except Exception as e:
        print(f"❌ Error getting customer details: {e}")
        print("Note: This would work with a live database connection.")
    
    print("\n" + "=" * 60)
    print("📝 Step 3: Create AI-powered preparation summary for staff")
    print("-" * 50)
    
    # Create AI-powered preparation summary
    create_summary = customer_tools["create_customer_preparation_summary"]
    
    print("Manager: 'Create a preparation summary for Sarah Johnson's personal shopping appointment'")
    print()
    
    try:
        summary = create_summary.invoke({
            "customer_name": "Sarah Johnson",
            "context": "personal shopping appointment"
        })
        
        print("✅ AI-Generated Customer Preparation Summary:")
        print("-" * 50)
        print(summary)
            
    except Exception as e:
        print(f"❌ Error creating AI summary: {e}")
        print("Note: This would work with a live database connection and LLM access.")
        
        # Show what the AI summary would look like
        print("\n📋 Example AI-Generated Summary:")
        print("-" * 40)
        print("""
**Customer Preparation Summary: Sarah Johnson**

## 👤 Customer Overview
- **Name**: Sarah Johnson (prefers: Sarah)
- **Tier**: Premium customer (5+ years)
- **Value**: $45,600 lifetime spend | $1,140 avg transaction
- **Satisfaction**: 4.9/5.0 (exceptional history)

## 📅 Appointment Context
- **When**: Today at 2:00 PM (3.2 hours away)
- **Type**: Personal Styling Session
- **Purpose**: Holiday party wardrobe refresh
- **Context**: Professional needs elegant cocktail attire for company events

## 🎯 Key Preparation Points
- 🤝 **Manager greeting required** - Premium customer protocol
- 👗 **Ensure Isabella Rodriguez is available** - Preferred stylist
- 🏠 **Set up private styling room** - Customer preference
- 🥗 **Prepare gluten-free refreshments** - Dietary restriction
- 💎 **Focus on versatile, investment pieces** - Customer style preference

## 🔧 Service Requirements
- **Styling Approach**: Classic professional with modern touches
- **Budget Focus**: $800-1500 range (comfortable spending level)
- **Color Palette**: Navy, cream, camel, soft pastels (avoid bright/neon)
- **Brand Focus**: Theory, Eileen Fisher, St. John, store premium brands
- **Fit Notes**: Petite length needed, prefers slightly loose blazers

## 💬 Conversation Starters
- Ask about upcoming company holiday events and dress codes
- Reference her excellent style choices from previous visits
- Mention new arrivals in her preferred brands
- Discuss versatile pieces that work for multiple occasions

## 🌟 Success Factors
- **Efficiency**: Customer is always punctual, appreciates organized service
- **Completeness**: Likes to see complete outfits styled together
- **Quality**: Focus on timeless investment pieces over trendy items
- **Personalization**: Remember her preference for classic professional style
- **Follow-up**: Document any new preferences for future visits

**Expected Outcome**: Successful styling session with 2-3 complete outfits for holiday events, maintaining her 4.9/5.0 satisfaction score.
""")
    
    print("\n" + "=" * 60)
    print("💡 Enhanced Manager Benefits with AI")
    print("-" * 50)
    print("With this AI-powered customer preparation system, managers can:")
    print()
    print("🧠 Generate intelligent, context-aware preparation summaries")
    print("🎯 Get specific, actionable preparation points for each customer")
    print("💬 Receive conversation starters and relationship-building tips")
    print("📊 Access comprehensive customer insights in an easy-to-read format")
    print("🔄 Adapt summaries for different appointment types and contexts")
    print("⚡ Save time with automated, professional staff briefings")
    print("🎨 Get styling and product recommendations tailored to each customer")
    print("🌟 Identify key success factors for exceptional service delivery")
    print()
    print("The AI understands context and generates summaries that help staff")
    print("deliver personalized, memorable experiences for every customer.")

if __name__ == "__main__":
    demonstrate_customer_preparation_workflow() 