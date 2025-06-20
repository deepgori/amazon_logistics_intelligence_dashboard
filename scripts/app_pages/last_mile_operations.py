# pages/last_mile_operations.py

import streamlit as st
import pandas as pd
import altair as alt

def render_page(df_last_mile_ops):
    """Renders the Last-Mile Operations page"""
    
    st.header("🚚 Last-Mile Operational Intelligence")
    st.write("*(Real-time insights on route optimization, vehicle efficiency, and delivery performance)*")
    
    if not df_last_mile_ops.empty:
        # Enhanced header with KPIs
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            avg_deliveries = df_last_mile_ops['num_deliveries'].mean()
            st.metric("Avg Deliveries per Route", f"{avg_deliveries:.1f}")
        with col2:
            avg_distance = df_last_mile_ops['actual_route_distance_km'].mean()
            st.metric("Avg Route Distance", f"{avg_distance:.1f} km")
        with col3:
            avg_duration = df_last_mile_ops['actual_route_duration_hours'].mean()
            st.metric("Avg Route Duration", f"{avg_duration:.1f} hrs")
        with col4:
            efficiency_score = (df_last_mile_ops['route_score'] == 'High').mean() * 100
            st.metric("High Efficiency Routes", f"{efficiency_score:.1f}%")

        # First visualization: Route Score Distribution
        st.subheader("📊 Route Efficiency Distribution")
        route_score_counts = df_last_mile_ops['route_score'].value_counts().reset_index(name='count')
        route_score_counts.columns = ['route_score', 'count']
        
        chart_route_score = alt.Chart(route_score_counts).mark_arc().encode(
            theta=alt.Theta(field="count", type="quantitative"),
            color=alt.Color("route_score", scale=alt.Scale(domain=['High', 'Medium', 'Low'], 
                                                        range=['#00D4AA', '#FFB84D', '#FF6B6B'])), 
            order=alt.Order("count", sort="descending"),
            tooltip=["route_score", "count", alt.Tooltip("count", format=".1%")]
        ).properties(
            title='Route Efficiency Distribution',
            height=300
        )
        st.altair_chart(chart_route_score, use_container_width=True)

        # Second visualization: Route Performance Analysis
        st.subheader("🎯 Route Performance Analysis")
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            # Distance vs Duration scatter plot
            scatter_chart = alt.Chart(df_last_mile_ops).mark_circle(size=60).encode(
                x=alt.X('actual_route_distance_km:Q', title='Route Distance (km)'),
                y=alt.Y('actual_route_duration_hours:Q', title='Route Duration (hours)'),
                color=alt.Color('route_score:N', scale=alt.Scale(domain=['High', 'Medium', 'Low'], 
                                                               range=['#00D4AA', '#FFB84D', '#FF6B6B'])),
                tooltip=['actual_route_distance_km', 'actual_route_duration_hours', 'route_score', 'num_deliveries']
            ).properties(
                title='Distance vs Duration Efficiency',
                height=300
            ).interactive()
            st.altair_chart(scatter_chart, use_container_width=True)
        
        with col_chart2:
            # Deliveries vs Volume efficiency
            volume_efficiency = alt.Chart(df_last_mile_ops).mark_circle(size=60).encode(
                x=alt.X('num_deliveries:Q', title='Number of Deliveries'),
                y=alt.Y('total_calculated_volume_cm3:Q', title='Total Volume (cm³)'),
                color=alt.Color('route_score:N', scale=alt.Scale(domain=['High', 'Medium', 'Low'], 
                                                               range=['#00D4AA', '#FFB84D', '#FF6B6B'])),
                tooltip=['num_deliveries', 'total_calculated_volume_cm3', 'route_score', 'actual_route_distance_km']
            ).properties(
                title='Delivery Volume Efficiency',
                height=300
            ).interactive()
            st.altair_chart(volume_efficiency, use_container_width=True)

        # Third visualization: Operational Insights Dashboard
        st.subheader("📈 Operational Insights & Optimization Opportunities")
        
        # Calculate key metrics
        route_insights = df_last_mile_ops.groupby('route_score').agg({
            'num_deliveries': ['mean', 'count'],
            'actual_route_distance_km': 'mean',
            'actual_route_duration_hours': 'mean',
            'total_calculated_volume_cm3': 'mean'
        }).round(2)
        
        # Flatten column names
        route_insights.columns = ['avg_deliveries', 'route_count', 'avg_distance', 'avg_duration', 'avg_volume']
        
        # Display insights in a more meaningful way
        col_insight1, col_insight2 = st.columns(2)
        
        with col_insight1:
            st.markdown("**🔍 Efficiency Analysis by Route Score**")
            for score in ['High', 'Medium', 'Low']:
                if score in route_insights.index:
                    data = route_insights.loc[score]
                    color = "#00D4AA" if score == 'High' else "#FFB84D" if score == 'Medium' else "#FF6B6B"
                    st.markdown(f"""
                    <div style="background: {color}20; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid {color};">
                        <h4 style="margin: 0 0 0.5rem 0; color: {color};">{score} Efficiency Routes</h4>
                        <p style="margin: 0.25rem 0;"><strong>{int(data['route_count'])} routes</strong> • Avg {data['avg_deliveries']:.1f} deliveries</p>
                        <p style="margin: 0.25rem 0;">{data['avg_distance']:.1f} km • {data['avg_duration']:.1f} hours</p>
                        <p style="margin: 0.25rem 0;">Volume: {data['avg_volume']:,.0f} cm³</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col_insight2:
            st.markdown("**💡 Optimization Recommendations**")
            
            # Calculate optimization opportunities
            low_efficiency_routes = df_last_mile_ops[df_last_mile_ops['route_score'] == 'Low']
            high_efficiency_routes = df_last_mile_ops[df_last_mile_ops['route_score'] == 'High']
            
            if not low_efficiency_routes.empty and not high_efficiency_routes.empty:
                # Distance optimization potential
                avg_low_distance = low_efficiency_routes['actual_route_distance_km'].mean()
                avg_high_distance = high_efficiency_routes['actual_route_distance_km'].mean()
                distance_savings = (avg_low_distance - avg_high_distance) * len(low_efficiency_routes)
                
                # Time optimization potential
                avg_low_duration = low_efficiency_routes['actual_route_duration_hours'].mean()
                avg_high_duration = high_efficiency_routes['actual_route_duration_hours'].mean()
                time_savings = (avg_low_duration - avg_high_duration) * len(low_efficiency_routes)
                
                st.markdown(f"""
                <div style="background: #6366F120; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid #6366F1;">
                    <h4 style="margin: 0 0 0.5rem 0; color: #6366F1;">🚀 Optimization Potential</h4>
                    <p style="margin: 0.25rem 0;"><strong>{len(low_efficiency_routes)} routes</strong> need optimization</p>
                    <p style="margin: 0.25rem 0;">Potential distance savings: <strong>{distance_savings:.1f} km/day</strong></p>
                    <p style="margin: 0.25rem 0;">Potential time savings: <strong>{time_savings:.1f} hours/day</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Enhanced Action Items section
                st.markdown("""
                    <div style="margin-top: 1.5rem;">
                        <h4 style="color: #FFFFFF; display: flex; align-items: center;">
                            <span style="font-size: 1.2rem; margin-right: 0.5rem;">🎯</span>
                            Action Items
                        </h4>
                    </div>
                """, unsafe_allow_html=True)
                
                action_items = [
                    {
                        "icon": "🔄",
                        "title": "Route Consolidation",
                        "description": "Combine nearby deliveries to reduce empty miles and improve vehicle utilization."
                    },
                    {
                        "icon": "⚡️",
                        "title": "Dynamic Routing",
                        "description": "Implement real-time route optimization based on live traffic data to avoid delays."
                    },
                    {
                        "icon": "🚚",
                        "title": "Vehicle Assignment",
                        "description": "Match vehicle capacity to route volume requirements to prevent underutilization or overloading."
                    },
                    {
                        "icon": "⏱️",
                        "title": "Time Windows",
                        "description": "Optimize delivery time slots for better efficiency and improved customer satisfaction."
                    }
                ]

                for item in action_items:
                    st.markdown(f"""
                    <div style="background: #242A30; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid #6366F1;">
                        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                             <span style="font-size: 1.5rem; margin-right: 0.75rem;">{item['icon']}</span>
                            <strong style="color: #FFFFFF; font-size: 1.1rem;">{item['title']}</strong>
                        </div>
                        <p style="margin: 0 0 0 2.25rem; color: rgba(255, 255, 255, 0.85); font-size: 0.95rem;">{item['description']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Insufficient data for optimization analysis")

        # Fourth visualization: Trend Analysis
        st.subheader("📊 Performance Trends & Patterns")
        
        # Create performance distribution charts
        col_trend1, col_trend2 = st.columns(2)
        
        with col_trend1:
            # Distance distribution by efficiency
            distance_dist = alt.Chart(df_last_mile_ops).mark_bar().encode(
                x=alt.X('actual_route_distance_km:Q', bin=alt.Bin(maxbins=15), title='Route Distance (km)'),
                y=alt.Y('count():Q', title='Number of Routes'),
                color=alt.Color('route_score:N', scale=alt.Scale(domain=['High', 'Medium', 'Low'], 
                                                               range=['#00D4AA', '#FFB84D', '#FF6B6B'])),
                tooltip=[alt.Tooltip('actual_route_distance_km', bin=True), 'count()', 'route_score']
            ).properties(
                title='Distance Distribution by Efficiency',
                height=250
            ).interactive()
            st.altair_chart(distance_dist, use_container_width=True)
        
        with col_trend2:
            # Duration distribution by efficiency
            duration_dist = alt.Chart(df_last_mile_ops).mark_bar().encode(
                x=alt.X('actual_route_duration_hours:Q', bin=alt.Bin(maxbins=15), title='Route Duration (hours)'),
                y=alt.Y('count():Q', title='Number of Routes'),
                color=alt.Color('route_score:N', scale=alt.Scale(domain=['High', 'Medium', 'Low'], 
                                                               range=['#00D4AA', '#FFB84D', '#FF6B6B'])),
                tooltip=[alt.Tooltip('actual_route_duration_hours', bin=True), 'count()', 'route_score']
            ).properties(
                title='Duration Distribution by Efficiency',
                height=250
            ).interactive()
            st.altair_chart(duration_dist, use_container_width=True)

    else:
        st.warning("🚚 Last-Mile Operations data is not available. Please check data generation and enhancement.") 