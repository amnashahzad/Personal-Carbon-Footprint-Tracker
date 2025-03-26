import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import folium
from streamlit_folium import folium_static
from datetime import datetime, date
import pytesseract  # Only for OCR (optional)

# Set page config
st.set_page_config(
    page_title="Carbon Footprint Tracker", 
    layout="wide",
    page_icon="🌍"
)

# Initialize session state with sample data
def initialize_data():
    if 'transactions' not in st.session_state:
        sample_transactions = pd.DataFrame({
            'Date': [date(2023, 1, 1), date(2023, 1, 2)],
            'Category': ['meat', 'fuel'],
            'Amount': [50, 30],
            'CO2_kg': [250, 75]
        })
        st.session_state.transactions = sample_transactions

    if 'locations' not in st.session_state:
        sample_locations = pd.DataFrame({
            'Date': [date(2023, 1, 1)],
            'Latitude': [37.7749],
            'Longitude': [-122.4194],
            'Transport': ['car'],
            'Distance_km': [15],
            'CO2_kg': [3.0]
        })
        st.session_state.locations = sample_locations

    if 'diet' not in st.session_state:
        sample_diet = pd.DataFrame({
            'Date': [date(2023, 1, 1)],
            'Food_Type': ['beef'],
            'Quantity': [1],
            'CO2_kg': [27.0]
        })
        st.session_state.diet = sample_diet

initialize_data()

# Carbon emission factors (kg CO2 per unit)
EMISSION_FACTORS = {
    'transport': {
        'car': 0.2,  # per km
        'bus': 0.1,
        'train': 0.05,
        'plane': 0.25,
        'walking': 0.0,
        'bicycle': 0.0
    },
    'spending': {
        'meat': 5.0,  # per $ spent
        'dairy': 2.0,
        'veggies': 0.5,
        'processed': 3.0,
        'fuel': 2.5,
        'electronics': 10.0,
        'clothing': 5.0,
        'other': 1.0
    },
    'food': {
        'beef': 27.0,  # per kg
        'chicken': 6.9,
        'fish': 5.1,
        'eggs': 4.5,
        'vegetables': 2.0,
        'fruits': 1.1,
        'dairy': 2.5,
        'other': 3.0
    }
}

def calculate_transaction_emissions(category, amount):
    """Calculate CO2 for spending category"""
    return amount * EMISSION_FACTORS['spending'].get(category.lower(), 1.0)

def calculate_food_emissions(food_type, quantity=1):
    """Calculate CO2 for food items"""
    return quantity * EMISSION_FACTORS['food'].get(food_type.lower(), 2.0)

def calculate_transport_emissions(distance_km, transport_type):
    """Calculate CO2 for transportation"""
    return distance_km * EMISSION_FACTORS['transport'].get(transport_type.lower(), 0.15)

def plot_emissions_trend():
    """Plot emissions over time with proper error handling"""
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Convert dates to datetime for proper grouping
        transactions = st.session_state.transactions.copy()
        locations = st.session_state.locations.copy()
        diet = st.session_state.diet.copy()
        
        transactions['Date'] = pd.to_datetime(transactions['Date'])
        locations['Date'] = pd.to_datetime(locations['Date'])
        diet['Date'] = pd.to_datetime(diet['Date'])
        
        # Combine all emissions by date
        all_emissions = pd.concat([
            transactions.groupby('Date')['CO2_kg'].sum(),
            locations.groupby('Date')['CO2_kg'].sum(),
            diet.groupby('Date')['CO2_kg'].sum()
        ], axis=1).fillna(0)
        
        if all_emissions.empty:
            st.warning("No emissions data available to plot. Add some data first!")
            return pd.DataFrame()
        
        all_emissions.columns = ['Spending', 'Transport', 'Diet']
        all_emissions['Total'] = all_emissions.sum(axis=1)
        all_emissions = all_emissions.sort_index()
        all_emissions.index = all_emissions.index.date
        
        all_emissions.plot(ax=ax)
        ax.set_title('Your Carbon Footprint Over Time')
        ax.set_ylabel('CO2 Emissions (kg)')
        ax.grid(True)
        st.pyplot(fig)
        
        return all_emissions
        
    except Exception as e:
        st.error(f"Error plotting emissions: {e}")
        return pd.DataFrame()

def show_emissions_map():
    """Show locations on map with emissions data"""
    try:
        if not st.session_state.locations.empty:
            center_lat = st.session_state.locations['Latitude'].mean()
            center_lon = st.session_state.locations['Longitude'].mean()
            
            m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
            
            for _, row in st.session_state.locations.iterrows():
                folium.CircleMarker(
                    location=[row['Latitude'], row['Longitude']],
                    radius=min(row['CO2_kg'] * 2, 20),
                    popup=f"{row['Transport']}: {row['CO2_kg']:.1f} kg CO2",
                    color='red',
                    fill=True,
                    fill_color='red'
                ).add_to(m)
            
            folium_static(m, width=800)
        else:
            st.info("No location data available. Add transportation data to see the map.")
            
    except Exception as e:
        st.error(f"Error generating map: {e}")

# App layout
st.title("🌱 Personal Carbon Footprint Tracker")
st.markdown("Track your carbon emissions from spending, transportation, and diet")

# Sidebar for data input
with st.sidebar:
    st.header("Add New Data")
    
    # Spending/Transactions
    with st.expander("💳 Add Spending"):
        input_date = st.date_input("Date", key="spending_date")
        category = st.selectbox(
            "Category",
            options=list(EMISSION_FACTORS['spending'].keys()),
            key="spending_category"
        )
        amount = st.number_input("Amount ($)", min_value=0.0, key="spending_amount")
        if st.button("Add Transaction", key="add_transaction"):
            co2 = calculate_transaction_emissions(category, amount)
            new_row = {
                'Date': input_date,
                'Category': category,
                'Amount': amount,
                'CO2_kg': co2
            }
            new_df = pd.DataFrame([new_row])
            st.session_state.transactions = pd.concat([
                st.session_state.transactions,
                new_df
            ], ignore_index=True)
            st.success("Transaction added!")
    
    # Location/Transport
    with st.expander("🚗 Add Transportation"):
        input_date = st.date_input("Travel Date", key="transport_date")
        transport = st.selectbox(
            "Transport Type",
            options=list(EMISSION_FACTORS['transport'].keys()),
            key="transport_type"
        )
        distance = st.number_input("Distance (km)", min_value=0.0, key="transport_distance")
        col1, col2 = st.columns(2)
        with col1:
            lat = st.number_input("Latitude", value=37.7749, key="transport_lat")
        with col2:
            lon = st.number_input("Longitude", value=-122.4194, key="transport_lon")
        if st.button("Add Transportation", key="add_transport"):
            co2 = calculate_transport_emissions(distance, transport)
            new_row = {
                'Date': input_date,
                'Latitude': lat,
                'Longitude': lon,
                'Transport': transport,
                'Distance_km': distance,
                'CO2_kg': co2
            }
            new_df = pd.DataFrame([new_row])
            st.session_state.locations = pd.concat([
                st.session_state.locations,
                new_df
            ], ignore_index=True)
            st.success("Transport data added!")
    
    # Diet/Food (without OCR for Streamlit Sharing)
    with st.expander("🍎 Add Diet Data"):
        input_date = st.date_input("Meal Date", key="diet_date")
        food_type = st.selectbox(
            "Food Type",
            options=list(EMISSION_FACTORS['food'].keys()),
            key="food_type"
        )
        quantity = st.number_input("Quantity", min_value=1, value=1, key="food_quantity")
        if st.button("Add Diet Data", key="add_diet"):
            co2 = calculate_food_emissions(food_type, quantity)
            new_row = {
                'Date': input_date,
                'Food_Type': food_type,
                'Quantity': quantity,
                'CO2_kg': co2
            }
            new_df = pd.DataFrame([new_row])
            st.session_state.diet = pd.concat([
                st.session_state.diet,
                new_df
            ], ignore_index=True)
            st.success("Diet data added!")

# Main dashboard
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📋 Data", "💡 Recommendations"])

with tab1:
    st.header("Your Carbon Footprint Dashboard")
    
    total_co2 = (
        st.session_state.transactions['CO2_kg'].sum() +
        st.session_state.locations['CO2_kg'].sum() +
        st.session_state.diet['CO2_kg'].sum()
    )
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total CO₂ Emissions", f"{total_co2:.1f} kg")
    col2.metric("Daily Average", f"{total_co2/max(1, len(st.session_state.transactions)):.1f} kg/day")
    col3.metric("Equivalent Trees Needed", f"{total_co2/21:.0f} trees/year")
    
    st.subheader("Emissions Over Time")
    emissions_data = plot_emissions_trend()
    
    st.subheader("Breakdown by Category")
    if not emissions_data.empty:
        fig, ax = plt.subplots(figsize=(8, 8))
        emissions_data[['Spending', 'Transport', 'Diet']].sum().plot.pie(
            autopct='%1.1f%%',
            ax=ax,
            colors=['#ff9999','#66b3ff','#99ff99']
        )
        ax.set_ylabel('')
        st.pyplot(fig)
    
    st.subheader("Transportation Map")
    show_emissions_map()

with tab2:
    st.header("Your Raw Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Spending Data")
        st.dataframe(
            st.session_state.transactions,
            column_config={
                "Date": "Date",
                "Category": "Category",
                "Amount": st.column_config.NumberColumn("Amount ($)", format="$%.2f"),
                "CO2_kg": st.column_config.NumberColumn("CO₂ (kg)", format="%.1f")
            },
            hide_index=True
        )
    
    with col2:
        st.subheader("Transportation Data")
        st.dataframe(
            st.session_state.locations,
            column_config={
                "Date": "Date",
                "Transport": "Transport",
                "Distance_km": st.column_config.NumberColumn("Distance (km)", format="%.1f"),
                "CO2_kg": st.column_config.NumberColumn("CO₂ (kg)", format="%.1f")
            },
            hide_index=True
        )
    
    with col3:
        st.subheader("Diet Data")
        st.dataframe(
            st.session_state.diet,
            column_config={
                "Date": "Date",
                "Food_Type": "Food Type",
                "Quantity": "Quantity",
                "CO2_kg": st.column_config.NumberColumn("CO₂ (kg)", format="%.1f")
            },
            hide_index=True
        )

with tab3:
    st.header("Personalized Recommendations")
    
    if total_co2 > 0:
        st.subheader("Top Ways to Reduce Your Footprint")
        
        if not st.session_state.transactions.empty:
            top_spending = st.session_state.transactions.groupby('Category')['CO2_kg'].sum().nlargest(1)
            for category, co2 in top_spending.items():
                st.write(f"🔸 **{category.capitalize()} Spending**: Creates {co2:.1f} kg CO₂")
                if category == 'meat':
                    st.write("- Try having 2 meat-free days each week (could save ~40% of food emissions)")
                elif category == 'fuel':
                    st.write("- Consider carpooling or public transport (could reduce transport emissions by 50%)")
        
        if not st.session_state.locations.empty:
            main_transport = st.session_state.locations['Transport'].mode()[0]
            if main_transport == 'car':
                st.write("🔸 **Car Usage**: Try alternatives for short trips:")
                st.write("- Walking or cycling for trips <3 km (zero emissions)")
                st.write("- Public transport for longer trips (emissions typically 50-75% lower than driving)")
        
        if not st.session_state.diet.empty:
            if 'beef' in st.session_state.diet['Food_Type'].values:
                st.write("🔸 **Beef Consumption**: The highest-emission food")
                st.write("- Replace beef with chicken (saves ~75% emissions)")
                st.write("- Try plant-based alternatives (saves ~90% emissions)")
            
            st.write("🔸 **General Food Tips**:")
            st.write("- Reduce food waste (30% of food is wasted globally)")
            st.write("- Buy seasonal local produce (lowers transport emissions)")
        
        st.subheader("Your Sustainability Goal")
        st.write(f"Current footprint: {total_co2:.1f} kg CO₂")
        st.write(f"Target: Try reducing by 20% to {(total_co2 * 0.8):.1f} kg CO₂")
        
    else:
        st.info("Add some data to get personalized recommendations")

# Add footer
st.markdown("---")
st.caption("🌱 Carbon Footprint Tracker v1.0 | Made with Streamlit")
