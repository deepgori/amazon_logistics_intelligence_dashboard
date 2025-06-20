# pages/cost_efficiency_analysis.py

import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

def render_page(df_orders_enhanced):
    """Renders the simplified Cost Efficiency Analysis page"""
    
    st.header("💰 Cost Efficiency Analysis")
    st.write("*(Essential cost analysis and optimization insights)*")
    
    if df_orders_enhanced.empty:
        st.warning("🚚 No data available for cost analysis.")
        return
    
    # Date range filter only
    if 'order_date' in df_orders_enhanced.columns:
        min_date = df_orders_enhanced['order_date'].min()
        max_date = df_orders_enhanced['order_date'].max()
        date_range = st.date_input(
            "📅 Time Period",
            value=(min_date.date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date()
        )
    else:
        date_range = None
    
    # Filter data
    filtered_df = df_orders_enhanced.copy()
    
    if date_range and len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df['order_date'] >= pd.Timestamp(date_range[0])) &
            (filtered_df['order_date'] <= pd.Timestamp(date_range[1]))
        ]
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_cost = filtered_df['delivery_cost_to_amazon'].sum()
        st.metric("Total Cost", f"${total_cost:,.2f}")
    
    with col2:
        avg_cost = filtered_df['delivery_cost_to_amazon'].mean()
        st.metric("Avg Cost/Order", f"${avg_cost:.2f}")
    
    with col3:
        high_cost_orders = len(filtered_df[filtered_df['delivery_cost_to_amazon'] > 10.0])
        total_orders = len(filtered_df)
        high_cost_pct = (high_cost_orders / total_orders * 100) if total_orders > 0 else 0
        st.metric("High-Cost Orders", f"{high_cost_pct:.1f}%")
    
    with col4:
        if 'carrier' in filtered_df.columns:
            best_carrier = filtered_df.groupby('carrier')['delivery_cost_to_amazon'].mean().idxmin()
            best_cost = filtered_df.groupby('carrier')['delivery_cost_to_amazon'].mean().min()
            st.metric("Best Carrier", f"{best_carrier} (${best_cost:.2f})")
        else:
            st.metric("Best Cost", f"${avg_cost:.2f}")
    
    # Cost distribution chart
    st.subheader("📊 Cost Distribution")
    cost_dist_chart = alt.Chart(filtered_df).mark_bar().encode(
        x=alt.X('delivery_cost_to_amazon:Q', bin=alt.Bin(maxbins=15), title='Delivery Cost ($)'),
        y=alt.Y('count():Q', title='Number of Orders'),
        color=alt.value('#FF9900'),
        tooltip=[alt.Tooltip('delivery_cost_to_amazon', bin=True), 'count()']
    ).properties(
        title='Cost Distribution',
        height=300
    ).interactive()
    
    st.altair_chart(cost_dist_chart, use_container_width=True)
    
    # Carrier cost comparison
    if 'carrier' in filtered_df.columns:
        st.subheader("🚚 Carrier Cost Comparison")
        
        carrier_costs = filtered_df.groupby('carrier').agg({
            'delivery_cost_to_amazon': ['mean', 'count'],
            'delivery_status': lambda x: (x == 'On-Time').mean() * 100
        }).round(2)
        
        carrier_costs.columns = ['Avg Cost', 'Orders', 'On-Time %']
        carrier_costs = carrier_costs.sort_values('Avg Cost')
        
        # Display as cards
        cols = st.columns(len(carrier_costs))
        for i, (carrier, data) in enumerate(carrier_costs.iterrows()):
            with cols[i]:
                color = "#28a745" if data['Avg Cost'] <= avg_cost else "#dc3545"
                st.markdown(f"""
                <div style="background: {color}20; padding: 1rem; border-radius: 8px; border-left: 4px solid {color};">
                    <h4 style="margin: 0 0 0.5rem 0; color: {color};">{carrier}</h4>
                    <p style="margin: 0.25rem 0;"><strong>Avg Cost:</strong> ${data['Avg Cost']:.2f}</p>
                    <p style="margin: 0.25rem 0;"><strong>Orders:</strong> {int(data['Orders']):,}</p>
                    <p style="margin: 0.25rem 0;"><strong>On-Time:</strong> {data['On-Time %']:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Cost optimization opportunities
    st.subheader("🎯 Cost Optimization Opportunities")
    opportunities = []
    
    # High cost carrier analysis
    if 'carrier' in filtered_df.columns:
        carrier_costs = filtered_df.groupby('carrier')['delivery_cost_to_amazon'].mean().sort_values(ascending=False)
        if len(carrier_costs) > 1:
            expensive_carrier = carrier_costs.index[0]
            cheap_carrier = carrier_costs.index[-1]
            cost_diff = carrier_costs.iloc[0] - carrier_costs.iloc[-1]
            
            if cost_diff > 2.0:
                potential_savings = cost_diff * len(filtered_df[filtered_df['carrier'] == expensive_carrier])
                opportunities.append(f"🚚 **Switch from {expensive_carrier} to {cheap_carrier}**: Save ${potential_savings:,.2f}")
    
    # High cost orders
    high_cost_pct = (len(filtered_df[filtered_df['delivery_cost_to_amazon'] > 10.0]) / len(filtered_df) * 100)
    if high_cost_pct > 10:
        opportunities.append(f"💰 **Reduce high-cost orders**: {high_cost_pct:.1f}% of orders exceed $10.00")
    
    # Service type optimization
    if 'is_prime_member' in filtered_df.columns:
        prime_costs = filtered_df[filtered_df['is_prime_member']]['delivery_cost_to_amazon'].mean()
        standard_costs = filtered_df[~filtered_df['is_prime_member']]['delivery_cost_to_amazon'].mean()
        
        if abs(prime_costs - standard_costs) > 1.0:
            if prime_costs > standard_costs:
                opportunities.append(f"⚡ **Prime costs ${prime_costs - standard_costs:.2f} more than standard**")
            else:
                opportunities.append(f"📈 **Prime is ${standard_costs - prime_costs:.2f} cheaper - consider expansion**")
    
    if opportunities:
        for opportunity in opportunities:
            st.markdown(f"""
            <div style="background: #28a74520; padding: 0.75rem; border-radius: 6px; margin: 0.5rem 0; border-left: 4px solid #28a745;">
                <span style="color: rgba(255, 255, 255, 0.9);">{opportunity}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ No major cost optimization opportunities identified.") 