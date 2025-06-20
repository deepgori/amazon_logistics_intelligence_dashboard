# pages/prime_performance.py

import streamlit as st
import pandas as pd

def render_page(df_orders_enhanced):
    """Renders the Prime Performance page"""
    
    # Enhanced header with status indicators
    col_header1, col_header2, col_header3 = st.columns([2, 1, 1])
    with col_header1:
        st.markdown("""
            <div style="background: #242A30; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); border: 1px solid rgba(255, 255, 255, 0.1);">
                <h2 style="margin: 0 0 0.5rem 0; color: #FFFFFF;">📊 Prime & Standard Delivery Performance</h2>
                <p style="margin: 0; color: rgba(255, 255, 255, 0.85);">Comprehensive analysis of delivery metrics, carrier strategy, and geographical performance</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col_header2:
        if not df_orders_enhanced.empty:
            total_orders = len(df_orders_enhanced)
            st.markdown(f"""
                <div style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%); color: white; padding: 1rem; border-radius: 8px; text-align: center;">
                    <div style="font-size: 1.5rem; font-weight: bold;">{total_orders:,}</div>
                    <div style="font-size: 0.9rem;">Total Orders</div>
                </div>
            """, unsafe_allow_html=True)
    
    with col_header3:
        if not df_orders_enhanced.empty:
            prime_orders = len(df_orders_enhanced[df_orders_enhanced['is_prime_member'] == True])
            st.markdown(f"""
                <div style="background: linear-gradient(135deg, #FF9900 0%, #FF8C00 100%); color: white; padding: 1rem; border-radius: 8px; text-align: center;">
                    <div style="font-size: 1.5rem; font-weight: bold;">{prime_orders:,}</div>
                    <div style="font-size: 0.9rem;">Prime Orders</div>
                </div>
            """, unsafe_allow_html=True)
    
    st.write("")  # Spacing
    
    # Enhanced KPIs with better styling
    if not df_orders_enhanced.empty:
        st.markdown("""
            <div style="background: #242A30; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); margin: 1rem 0; border: 1px solid rgba(255, 255, 255, 0.1);">
                <h3 style="margin: 0 0 1rem 0; color: #FFFFFF;">📈 Key Performance Indicators</h3>
            </div>
        """, unsafe_allow_html=True)
        
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

        # KPI 1: Prime Delivery Days
        with kpi_col1:
            avg_prime_days = df_orders_enhanced[df_orders_enhanced['is_prime_member'] == True]['delivery_days_actual'].mean()
            st.markdown(f"""
                <div style="background: #242A30; padding: 1rem; border-radius: 8px; border-left: 5px solid #FF9900; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                        <span style="font-size: 1.5rem; margin-right: 0.75rem;">📦</span>
                        <h5 style="margin: 0; color: #FFFFFF;">Prime Delivery Days</h5>
                    </div>
                    <p style="font-size: 2rem; font-weight: 700; color: #FFFFFF; margin: 0;">{avg_prime_days:.1f} <span style="font-size: 1rem; font-weight: 400;">days</span></p>
                </div>
            """, unsafe_allow_html=True)

        # KPI 2: Standard Delivery Days
        with kpi_col2:
            avg_standard_days = df_orders_enhanced[df_orders_enhanced['is_prime_member'] == False]['delivery_days_actual'].mean()
            delta_value = avg_standard_days - avg_prime_days
            delta_color = "#dc3545" if delta_value > 0 else "#28a745"
            st.markdown(f"""
                <div style="background: #242A30; padding: 1rem; border-radius: 8px; border-left: 5px solid #17a2b8; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                        <span style="font-size: 1.5rem; margin-right: 0.75rem;">🐢</span>
                        <h5 style="margin: 0; color: #FFFFFF;">Standard Delivery Days</h5>
                    </div>
                    <p style="font-size: 2rem; font-weight: 700; color: #FFFFFF; margin: 0;">{avg_standard_days:.1f} <span style="font-size: 1rem; font-weight: 400;">days</span></p>
                    <p style="color: {delta_color}; font-weight: 600; margin: 0.25rem 0 0 0;">{delta_value:+.1f} days vs Prime</p>
                </div>
            """, unsafe_allow_html=True)

        # KPI 3: Prime On-Time %
        with kpi_col3:
            prime_on_time_pct = (df_orders_enhanced[df_orders_enhanced['is_prime_member'] == True]['delivery_status'] == 'On-Time').mean() * 100
            st.markdown(f"""
                <div style="background: #242A30; padding: 1rem; border-radius: 8px; border-left: 5px solid #FF9900; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                        <span style="font-size: 1.5rem; margin-right: 0.75rem;">✅</span>
                        <h5 style="margin: 0; color: #FFFFFF;">Prime On-Time %</h5>
                    </div>
                    <p style="font-size: 2rem; font-weight: 700; color: #FFFFFF; margin: 0;">{prime_on_time_pct:.1f}<span style="font-size: 1rem; font-weight: 400;">%</span></p>
                </div>
            """, unsafe_allow_html=True)

        # KPI 4: Standard On-Time %
        with kpi_col4:
            standard_on_time_pct = (df_orders_enhanced[df_orders_enhanced['is_prime_member'] == False]['delivery_status'] == 'On-Time').mean() * 100
            st.markdown(f"""
                <div style="background: #242A30; padding: 1rem; border-radius: 8px; border-left: 5px solid #17a2b8; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                        <span style="font-size: 1.5rem; margin-right: 0.75rem;">✔️</span>
                        <h5 style="margin: 0; color: #FFFFFF;">Standard On-Time %</h5>
                    </div>
                    <p style="font-size: 2rem; font-weight: 700; color: #FFFFFF; margin: 0;">{standard_on_time_pct:.1f}<span style="font-size: 1rem; font-weight: 400;">%</span></p>
                </div>
            """, unsafe_allow_html=True)

        # Enhanced charts section
        st.markdown("""
            <div style="background: #242A30; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); margin: 1rem 0; border: 1px solid rgba(255, 255, 255, 0.1);">
                <h3 style="margin: 0 0 1rem 0; color: #FFFFFF;">📊 Delivery Trends & Distribution</h3>
            </div>
        """, unsafe_allow_html=True)
        
        # Delivery distribution chart
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("**📈 Delivery Days Distribution**")
            delivery_dist = df_orders_enhanced.groupby(['delivery_days_actual', 'is_prime_member']).size().unstack(fill_value=0)
            # Rename columns for a user-friendly legend
            delivery_dist.rename(columns={True: 'Prime', False: 'Standard'}, inplace=True)
            st.bar_chart(delivery_dist, use_container_width=True)

        with col_chart2:
            st.markdown("**🚚 Carrier Distribution**")
            carrier_dist = df_orders_enhanced.groupby('carrier')['order_id'].count().sort_values(ascending=False)
            st.bar_chart(carrier_dist, use_container_width=True)

        # Enhanced Cost Analysis Dashboard
        st.markdown("""
            <div style="background: #242A30; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); margin: 1rem 0; border: 1px solid rgba(255, 255, 255, 0.1);">
                <h3 style="margin: 0 0 1rem 0; color: #FFFFFF;">💰 Cost Analysis & Optimization</h3>
            </div>
        """, unsafe_allow_html=True)
        
        # Cost breakdown by Prime vs Standard
        col_cost1, col_cost2 = st.columns(2)
        
        with col_cost1:
            st.markdown("**📊 Cost Breakdown by Service Type**")
            
            # Calculate cost metrics by Prime status
            cost_by_prime = df_orders_enhanced.groupby('is_prime_member').agg({
                'delivery_cost_to_amazon': ['sum', 'mean', 'count'],
                'delivery_status': lambda x: (x == 'On-Time').mean() * 100
            }).round(2)
            
            # Flatten column names
            cost_by_prime.columns = ['total_cost', 'avg_cost_per_order', 'order_count', 'on_time_rate']
            
            # Display Prime metrics
            prime_data = cost_by_prime.loc[True] if True in cost_by_prime.index else None
            standard_data = cost_by_prime.loc[False] if False in cost_by_prime.index else None
            
            if prime_data is not None:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #FF9900 0%, #FF8C00 100%); color: white; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">
                    <h4 style="margin: 0 0 0.5rem 0;">Prime Delivery</h4>
                    <p style="margin: 0.25rem 0;"><strong>Total Cost:</strong> ${prime_data['total_cost']:,.2f}</p>
                    <p style="margin: 0.25rem 0;"><strong>Avg Cost/Order:</strong> ${prime_data['avg_cost_per_order']:.2f}</p>
                    <p style="margin: 0.25rem 0;"><strong>Orders:</strong> {int(prime_data['order_count']):,}</p>
                    <p style="margin: 0.25rem 0;"><strong>On-Time Rate:</strong> {prime_data['on_time_rate']:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            if standard_data is not None:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #17a2b8 0%, #138496 100%); color: white; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">
                    <h4 style="margin: 0 0 0.5rem 0;">Standard Delivery</h4>
                    <p style="margin: 0.25rem 0;"><strong>Total Cost:</strong> ${standard_data['total_cost']:,.2f}</p>
                    <p style="margin: 0.25rem 0;"><strong>Avg Cost/Order:</strong> ${standard_data['avg_cost_per_order']:.2f}</p>
                    <p style="margin: 0.25rem 0;"><strong>Orders:</strong> {int(standard_data['order_count']):,}</p>
                    <p style="margin: 0.25rem 0;"><strong>On-Time Rate:</strong> {standard_data['on_time_rate']:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col_cost2:
            st.markdown("**📈 Cost Efficiency Analysis**")
            
            # Calculate cost efficiency metrics
            if prime_data is not None and standard_data is not None:
                # Cost per successful delivery
                prime_cost_per_success = prime_data['total_cost'] / (prime_data['order_count'] * prime_data['on_time_rate'] / 100)
                standard_cost_per_success = standard_data['total_cost'] / (standard_data['order_count'] * standard_data['on_time_rate'] / 100)
                
                # Cost savings potential
                if standard_data['avg_cost_per_order'] > prime_data['avg_cost_per_order']:
                    potential_savings = (standard_data['avg_cost_per_order'] - prime_data['avg_cost_per_order']) * standard_data['order_count']
                    savings_message = f"Potential savings: ${potential_savings:,.2f}"
                    savings_color = "#28a745"
                else:
                    potential_savings = 0
                    savings_message = "Standard delivery is cost-effective"
                    savings_color = "#17a2b8"
                
                st.markdown(f"""
                <div style="background: #1A1F24; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid {savings_color}; border: 1px solid rgba(255, 255, 255, 0.1);">
                    <h4 style="margin: 0 0 0.5rem 0; color: #FFFFFF;">Cost Efficiency Metrics</h4>
                    <p style="margin: 0.25rem 0; color: rgba(255, 255, 255, 0.85);"><strong>Prime Cost/Success:</strong> ${prime_cost_per_success:.2f}</p>
                    <p style="margin: 0.25rem 0; color: rgba(255, 255, 255, 0.85);"><strong>Standard Cost/Success:</strong> ${standard_cost_per_success:.2f}</p>
                    <p style="margin: 0.25rem 0; color: {savings_color};"><strong>{savings_message}</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Cost trend analysis
                recent_cost = df_orders_enhanced.sort_values('order_date').tail(100)['delivery_cost_to_amazon'].mean()
                historical_cost = df_orders_enhanced.sort_values('order_date').iloc[:-100]['delivery_cost_to_amazon'].mean()
                cost_trend = ((recent_cost - historical_cost) / historical_cost * 100) if historical_cost > 0 else 0
                
                trend_color = "#dc3545" if cost_trend > 5 else "#28a745" if cost_trend < -5 else "#ffc107"
                trend_icon = "📈" if cost_trend > 5 else "📉" if cost_trend < -5 else "➡️"
                
                st.markdown(f"""
                <div style="background: #1A1F24; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid {trend_color}; border: 1px solid rgba(255, 255, 255, 0.1);">
                    <h4 style="margin: 0 0 0.5rem 0; color: #FFFFFF;">Cost Trend Analysis</h4>
                    <p style="margin: 0.25rem 0; color: rgba(255, 255, 255, 0.85);"><strong>Recent Avg Cost:</strong> ${recent_cost:.2f}</p>
                    <p style="margin: 0.25rem 0; color: rgba(255, 255, 255, 0.85);"><strong>Historical Avg Cost:</strong> ${historical_cost:.2f}</p>
                    <p style="margin: 0.25rem 0; color: {trend_color};"><strong>{trend_icon} Cost Trend:</strong> {cost_trend:+.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Cost optimization insights
        st.markdown("**💡 Cost Optimization Insights**")
        col_insight1, col_insight2 = st.columns(2)
        
        with col_insight1:
            # Cost efficiency by customer value tier
            if 'customer_lifetime_value_tier' in df_orders_enhanced.columns:
                customer_tier_analysis = df_orders_enhanced.groupby('customer_lifetime_value_tier').agg({
                    'delivery_cost_to_amazon': ['mean', 'sum', 'count'],
                    'delivery_status': lambda x: (x == 'On-Time').mean() * 100
                }).round(2)
                customer_tier_analysis.columns = ['avg_cost', 'total_cost', 'order_count', 'on_time_rate']
                customer_tier_analysis['roi_score'] = customer_tier_analysis['on_time_rate'] / customer_tier_analysis['avg_cost']
                
                st.markdown("**👥 Customer Value Tier Analysis**")
                for tier in customer_tier_analysis.index:
                    data = customer_tier_analysis.loc[tier]
                    tier_color = "#28a745" if tier == 'High' else "#ffc107" if tier == 'Medium' else "#dc3545"
                    st.markdown(f"""
                    <div style="background: {tier_color}20; padding: 0.75rem; border-radius: 6px; margin: 0.25rem 0; border-left: 4px solid {tier_color};">
                        <strong style="color: #FFFFFF;">{tier} Value Customers</strong><br>
                        <span style="color: rgba(255, 255, 255, 0.85);">${data['avg_cost']:.2f}/order • {data['on_time_rate']:.1f}% on-time</span><br>
                        <span style="color: rgba(255, 255, 255, 0.85);">{int(data['order_count']):,} orders • ROI Score: {data['roi_score']:.2f}</span>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col_insight2:
            # Cost savings opportunities
            st.markdown("**💰 Cost Savings Opportunities**")
            
            savings_opportunities = []
            
            # Analyze carrier cost differences
            if 'carrier' in df_orders_enhanced.columns:
                carrier_costs = df_orders_enhanced.groupby('carrier')['delivery_cost_to_amazon'].mean().sort_values()
                if len(carrier_costs) > 1:
                    cost_diff = carrier_costs.iloc[-1] - carrier_costs.iloc[0]
                    expensive_carrier = carrier_costs.index[-1]
                    cheap_carrier = carrier_costs.index[0]
                    savings_opportunities.append(f"• Switch from {expensive_carrier} to {cheap_carrier}: ${cost_diff:.2f}/order savings")
            
            # Analyze Prime vs Standard optimization
            if prime_data is not None and standard_data is not None:
                if standard_data['avg_cost_per_order'] > prime_data['avg_cost_per_order']:
                    potential_savings = (standard_data['avg_cost_per_order'] - prime_data['avg_cost_per_order']) * standard_data['order_count']
                    savings_opportunities.append(f"• Expand Prime service: ${potential_savings:,.2f} potential savings")
            
            # Analyze volume optimization
            if len(df_orders_enhanced) > 0:
                avg_orders_per_day = len(df_orders_enhanced) / max(1, (df_orders_enhanced['order_date'].max() - df_orders_enhanced['order_date'].min()).days)
                if avg_orders_per_day < 100:
                    savings_opportunities.append("• Increase daily volume to 100+ orders for better economies of scale")
            
            # Analyze city-based optimization
            if 'destination_city' in df_orders_enhanced.columns:
                high_cost_cities = df_orders_enhanced.groupby('destination_city')['delivery_cost_to_amazon'].mean().nlargest(3)
                if not high_cost_cities.empty:
                    savings_opportunities.append(f"• Review delivery strategy for high-cost cities: {', '.join(high_cost_cities.index)}")
            
            if savings_opportunities:
                for opportunity in savings_opportunities:
                    st.markdown(f"""
                    <div style="background: #28a74520; padding: 0.5rem; border-radius: 6px; margin: 0.25rem 0; border-left: 4px solid #28a745;">
                        <span style="color: rgba(255, 255, 255, 0.85);">{opportunity}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("✅ No immediate cost savings opportunities identified")

        # Enhanced geographical overview
        st.markdown("""
            <div style="background: #242A30; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); margin: 1rem 0; border: 1px solid rgba(255, 255, 255, 0.1);">
                <h3 style="margin: 0 0 1rem 0; color: #FFFFFF;">🗺️ Geographical Overview</h3>
            </div>
        """, unsafe_allow_html=True)
        
        col_map, col_stats = st.columns([2, 1])
        
        with col_map:
            # Fix the map column names to match Streamlit's expected format
            if 'destination_latitude' in df_orders_enhanced.columns and 'destination_longitude' in df_orders_enhanced.columns:
                map_data = df_orders_enhanced[['destination_latitude', 'destination_longitude']].dropna().copy()
                map_data.rename(columns={'destination_latitude': 'lat', 'destination_longitude': 'lon'}, inplace=True)
                st.map(map_data)
            else:
                st.warning("Geographical data (latitude, longitude) is not available to display the map.")
        
        with col_stats:
            # Top performing cities
            if 'destination_city' in df_orders_enhanced.columns and 'delivery_status' in df_orders_enhanced.columns:
                city_performance = df_orders_enhanced.groupby('destination_city').agg({
                    'delivery_status': lambda x: (x == 'On-Time').mean() * 100,
                    'order_id': 'count'
                }).round(2)
                city_performance.columns = ['On-Time %', 'Order Count']
                city_performance = city_performance.sort_values('On-Time %', ascending=False)
                
                st.markdown("**🏆 Top Performing Cities**")
                for city in city_performance.head(5).index:
                    performance = city_performance.loc[city]
                    st.markdown(f"""
                    <div style="background: #1A1F24; padding: 0.5rem; border-radius: 6px; margin: 0.25rem 0; border: 1px solid rgba(255, 255, 255, 0.1);">
                        <strong style="color: #FFFFFF;">{city}</strong><br>
                        <span style="color: rgba(255, 255, 255, 0.85);">{performance['On-Time %']:.1f}% on-time • {int(performance['Order Count']):,} orders</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("City performance data is not available.")

    else:
        st.error("Prime Performance data is not available. Please check data generation and enhancement.") 