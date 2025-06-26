"""Geographical Analysis tab for VP of Retail Operations."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def generate_time_based_metrics(time_range):
    """Generate different metrics based on selected time range."""
    
    # Base metrics with time-specific variations
    metrics_data = {
        "Last 30 Days": {
            "total_revenue": {"value": "$47.8M", "delta": "+12.3%"},
            "total_stores": {"value": "342", "delta": "+8 stores"},
            "avg_performance": {"value": "105.9%", "delta": "+4.2%"},
            "customer_satisfaction": {"value": "4.7/5.0", "delta": "+0.3"},
            "operational_efficiency": {"value": "94.2%", "delta": "+2.1%"}
        },
        "Last Quarter": {
            "total_revenue": {"value": "$142.3M", "delta": "+18.7%"},
            "total_stores": {"value": "342", "delta": "+12 stores"},
            "avg_performance": {"value": "108.4%", "delta": "+6.8%"},
            "customer_satisfaction": {"value": "4.6/5.0", "delta": "+0.2"},
            "operational_efficiency": {"value": "96.1%", "delta": "+4.3%"}
        },
        "YTD": {
            "total_revenue": {"value": "$523.7M", "delta": "+22.1%"},
            "total_stores": {"value": "342", "delta": "+28 stores"},
            "avg_performance": {"value": "110.2%", "delta": "+8.9%"},
            "customer_satisfaction": {"value": "4.5/5.0", "delta": "+0.1"},
            "operational_efficiency": {"value": "97.8%", "delta": "+5.7%"}
        },
        "Last Year": {
            "total_revenue": {"value": "$1.89B", "delta": "+15.4%"},
            "total_stores": {"value": "314", "delta": "+42 stores"},
            "avg_performance": {"value": "107.6%", "delta": "+12.3%"},
            "customer_satisfaction": {"value": "4.4/5.0", "delta": "+0.4"},
            "operational_efficiency": {"value": "95.3%", "delta": "+8.1%"}
        }
    }
    
    return metrics_data.get(time_range, metrics_data["Last 30 Days"])


def generate_time_based_regional_data(time_range):
    """Generate different regional performance data based on selected time range."""
    
    # Base regional data with time-specific variations
    regional_data = {
        "Last 30 Days": {
            "West Coast": {"performance": 92.4, "revenue": 6.2, "stores": 58},
            "Pacific Northwest": {"performance": 107.2, "revenue": 6.1, "stores": 45},
            "Southwest": {"performance": 106.8, "revenue": 7.2, "stores": 54},
            "Mountain West": {"performance": 103.5, "revenue": 5.8, "stores": 42},
            "South Central": {"performance": 110.2, "revenue": 10.1, "stores": 73},
            "Midwest": {"performance": 98.2, "revenue": 9.2, "stores": 61},
            "Southeast": {"performance": 112.4, "revenue": 8.7, "stores": 68},
            "Northeast": {"performance": 104.7, "revenue": 11.8, "stores": 82},
            "Mid-Atlantic": {"performance": 95.8, "revenue": 5.7, "stores": 44},
        },
        "Last Quarter": {
            "West Coast": {"performance": 95.1, "revenue": 18.8, "stores": 58},
            "Pacific Northwest": {"performance": 109.8, "revenue": 18.2, "stores": 45},
            "Southwest": {"performance": 108.4, "revenue": 21.1, "stores": 54},
            "Mountain West": {"performance": 105.9, "revenue": 17.4, "stores": 42},
            "South Central": {"performance": 112.7, "revenue": 29.8, "stores": 73},
            "Midwest": {"performance": 100.1, "revenue": 27.3, "stores": 61},
            "Southeast": {"performance": 114.8, "revenue": 25.9, "stores": 68},
            "Northeast": {"performance": 107.2, "revenue": 34.7, "stores": 82},
            "Mid-Atlantic": {"performance": 98.3, "revenue": 16.9, "stores": 44},
        },
        "YTD": {
            "West Coast": {"performance": 97.8, "revenue": 68.4, "stores": 58},
            "Pacific Northwest": {"performance": 111.5, "revenue": 66.1, "stores": 45},
            "Southwest": {"performance": 110.9, "revenue": 76.8, "stores": 54},
            "Mountain West": {"performance": 108.2, "revenue": 63.2, "stores": 42},
            "South Central": {"performance": 115.3, "revenue": 108.7, "stores": 73},
            "Midwest": {"performance": 102.8, "revenue": 99.4, "stores": 61},
            "Southeast": {"performance": 117.1, "revenue": 94.3, "stores": 68},
            "Northeast": {"performance": 109.8, "revenue": 126.5, "stores": 82},
            "Mid-Atlantic": {"performance": 101.4, "revenue": 61.7, "stores": 44},
        },
        "Last Year": {
            "West Coast": {"performance": 101.2, "revenue": 248.9, "stores": 58},
            "Pacific Northwest": {"performance": 113.8, "revenue": 240.3, "stores": 45},
            "Southwest": {"performance": 112.4, "revenue": 279.2, "stores": 54},
            "Mountain West": {"performance": 110.7, "revenue": 229.8, "stores": 42},
            "South Central": {"performance": 118.1, "revenue": 395.2, "stores": 73},
            "Midwest": {"performance": 105.4, "revenue": 361.7, "stores": 61},
            "Southeast": {"performance": 119.6, "revenue": 343.1, "stores": 68},
            "Northeast": {"performance": 112.3, "revenue": 460.3, "stores": 82},
            "Mid-Atlantic": {"performance": 104.1, "revenue": 224.5, "stores": 44},
        }
    }
    
    return regional_data.get(time_range, regional_data["Last 30 Days"])


def show_geographical_analysis_tab():
    """Display geographical analysis with interactive drill-down capabilities."""

    # All selectors in one row - determine layout based on analysis level
    analysis_level = st.session_state.get("geo_analysis_level", "National Overview")
    time_range = st.session_state.get("geo_time_range", "Last 30 Days")
    
    # Create columns dynamically based on analysis level
    if analysis_level == "National Overview":
        # 2 columns: Time Range + Analysis Level
        col1, col2 = st.columns(2)
        columns = [col1, col2]
    elif analysis_level == "Regional Analysis":
        # 3 columns: Time Range + Analysis Level + Region
        col1, col2, col3 = st.columns(3)
        columns = [col1, col2, col3]
    elif analysis_level == "Market Analysis":
        # 4 columns: Time Range + Analysis Level + Region + Market
        col1, col2, col3, col4 = st.columns(4)
        columns = [col1, col2, col3, col4]
    elif analysis_level == "District Analysis":
        # 5 columns: Time Range + Analysis Level + Region + Market + District
        col1, col2, col3, col4, col5 = st.columns(5)
        columns = [col1, col2, col3, col4, col5]
    else:  # Store Analysis
        # 6 columns: Time Range + Analysis Level + Region + Market + District + Store
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        columns = [col1, col2, col3, col4, col5, col6]

    # Time Range (always in first column)
    with columns[0]:
        time_range = st.selectbox(
            "Time Range:",
            ["Last 30 Days", "Last Quarter", "YTD", "Last Year"],
            key="geo_time_range",
        )

    # Analysis Level (always in second column)
    with columns[1]:
        analysis_level = st.selectbox(
            "Analysis Level:",
            [
                "National Overview",
                "Regional Analysis",
                "Market Analysis", 
                "District Analysis",
                "Store Analysis",
            ],
            key="geo_analysis_level",
        )

    # Generate time-based data
    regions = generate_time_based_regional_data(time_range)

    # Dynamic selectors based on analysis level
    selected_region = None
    selected_market = None 
    selected_district = None
    selected_store = None

    if analysis_level in ["Regional Analysis", "Market Analysis", "District Analysis", "Store Analysis"]:
        # Region selector (third column)
        with columns[2]:
            selected_region = st.selectbox(
                "Select Region:",
                ["All Regions"] + list(regions.keys()),
                key="geo_selected_region",
            )

    if analysis_level in ["Market Analysis", "District Analysis", "Store Analysis"]:
        # Market selector (fourth column)
        with columns[3]:
            if selected_region and selected_region != "All Regions":
                markets = regions[selected_region]["markets"] if "markets" in regions[selected_region] else ["California", "Nevada"]
            else:
                markets = [
                    "All Markets",
                    "California",
                    "Washington", 
                    "Texas",
                    "New York",
                    "Florida", 
                    "Illinois",
                    "Arizona",
                    "Colorado",
                ]
            selected_market = st.selectbox(
                "Select Market:",
                markets,
                key="geo_selected_market"
            )

    if analysis_level in ["District Analysis", "Store Analysis"]:
        # District selector (fifth column)
        with columns[4]:
            districts = [
                "All Districts",
                "District A",
                "District B", 
                "District C",
                "District D",
            ]
            selected_district = st.selectbox(
                "Select District:",
                districts,
                key="geo_selected_district"
            )

    if analysis_level == "Store Analysis":
        # Store selector (sixth column)
        with columns[5]:
            stores = [
                "All Stores",
                "Store #101",
                "Store #102",
                "Store #103",
                "Store #104",
            ]
            selected_store = st.selectbox(
                "Select Store:",
                stores,
                key="geo_selected_store"
            )


    # Dynamic content based on analysis level
    if analysis_level == "National Overview":
        show_national_overview(regions, time_range)
    elif analysis_level == "Regional Analysis":
        if selected_region == "All Regions":
            show_all_regions_analysis(regions)
        else:
            show_single_region_analysis(selected_region, regions[selected_region])
    elif analysis_level == "Market Analysis":
        show_market_analysis()
    elif analysis_level == "District Analysis":
        show_district_analysis()
    else:  # Store Analysis
        show_store_analysis()


def show_national_overview(regions, time_range):
    """Display national-level overview with regional breakdown."""
    
    # Get time-based metrics
    metrics = generate_time_based_metrics(time_range)

    # National KPI Summary with dynamic data
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-header">
                <!-- Icon removed -->
            </div>
            <div class="metric-value-container">
                <div class="metric-value">{metrics["total_revenue"]["value"]}</div>
                <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">{metrics["total_revenue"]["delta"]}</div>
            </div>
            <div class="metric-label">Total Revenue</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-header">
                <!-- Icon removed -->
            </div>
            <div class="metric-value-container">
                <div class="metric-value">{metrics["total_stores"]["value"]}</div>
                <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">{metrics["total_stores"]["delta"]}</div>
            </div>
            <div class="metric-label">Total Stores</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-header">
                <!-- Icon removed -->
            </div>
            <div class="metric-value-container">
                <div class="metric-value">{metrics["avg_performance"]["value"]}</div>
                <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">{metrics["avg_performance"]["delta"]}</div>
            </div>
            <div class="metric-label">Avg Performance</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-header">
                <!-- Icon removed -->
            </div>
            <div class="metric-value-container">
                <div class="metric-value">{metrics["customer_satisfaction"]["value"]}</div>
                <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">{metrics["customer_satisfaction"]["delta"]}</div>
            </div>
            <div class="metric-label">Customer Satisfaction</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col5:
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-header">
                <!-- Icon removed -->
            </div>
            <div class="metric-value-container">
                <div class="metric-value">{metrics["operational_efficiency"]["value"]}</div>
                <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">{metrics["operational_efficiency"]["delta"]}</div>
            </div>
            <div class="metric-label">Operational Efficiency</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Regional Performance Map and Charts
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### Regional Performance Map")

        # Update coordinates for better geographical distribution and accuracy
        region_coordinates = {
            "West Coast": (37.7749, -122.4194),      # San Francisco, CA
            "Pacific Northwest": (47.6062, -122.3321), # Seattle, WA
            "Southwest": (33.4484, -112.0740),       # Phoenix, AZ
            "Mountain West": (39.7392, -104.9903),   # Denver, CO
            "South Central": (32.7767, -96.7970),    # Dallas, TX
            "Midwest": (41.8781, -87.6298),          # Chicago, IL
            "Southeast": (33.7490, -84.3880),        # Atlanta, GA
            "Northeast": (42.3601, -71.0589),        # Boston, MA
            "Mid-Atlantic": (39.2904, -76.6122),     # Baltimore, MD
        }

        # Create regional performance data with consistent coordinates
        regions_df = pd.DataFrame(
            [
                {
                    "Region": region,
                    "Performance": data["performance"],
                    "Revenue_M": data["revenue"],
                    "Store_Count": data["stores"],
                    "Latitude": region_coordinates[region][0],
                    "Longitude": region_coordinates[region][1],
                    "Size": data["revenue"] * 5,  # For bubble size
                }
                for region, data in regions.items()
            ]
        )

        # Create map visualization
        fig_map = px.scatter_geo(
            regions_df,
            lat="Latitude",
            lon="Longitude",
            size="Revenue_M",
            color="Performance",
            hover_name="Region",
            hover_data={
                "Revenue_M": ":$.1f",
                "Store_Count": True,
                "Performance": ":.1f%",
            },
            color_continuous_scale="RdYlGn",
            size_max=50,
        )

        fig_map.update_geos(scope="usa", showlakes=True, lakecolor="rgb(255, 255, 255)")

        fig_map.update_layout(
            height=400,
            margin=dict(l=0, r=0, t=40, b=0),
            paper_bgcolor='rgba(0,0,0,0)',  # Transparent background for dark mode
            plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot area for dark mode
            geo=dict(
                bgcolor='rgba(0,0,0,0)',  # Transparent geo background for dark mode
            )
        )

        st.plotly_chart(fig_map, use_container_width=True)

    with col2:
        st.markdown("#### Regional Performance Scores")
        
        # Sort regions by performance
        regions_sorted = sorted(
            regions.items(), key=lambda x: x[1]["performance"], reverse=True
        )
        
        # Create compact performance visualization
        performance_data = []
        for region, data in regions_sorted:
            performance_data.append({
                'Region': region,
                'Performance': data['performance'],
                'Revenue': data['revenue'],
                'Stores': data['stores']
            })
        
        perf_df = pd.DataFrame(performance_data)
        
        # Performance bar chart with color coding
        fig_performance = px.bar(
            perf_df,
            x='Performance',
            y='Region',
            orientation='h',
            color='Performance',
            color_continuous_scale='RdYlGn',
            range_color=[95, 115],
            text='Performance'
        )
        
        # Add reference line at 100%
        fig_performance.add_vline(x=100, line_dash="dash", line_color="gray", 
                                 annotation_text="Target", annotation_position="top")
        
        fig_performance.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_performance.update_layout(
            height=400,
            margin=dict(l=0, r=0, t=40, b=0),
            showlegend=False,
            yaxis={'categoryorder':'total ascending'}
        )
        
        st.plotly_chart(fig_performance, use_container_width=True)

    # Quick Alert Summary - moved below the map
    st.markdown("##### 🚨 Quick Alerts")
    
    # Sort regions by performance for alerts
    regions_sorted = sorted(
        regions.items(), key=lambda x: x[1]["performance"], reverse=True
    )
    
    top_performer = regions_sorted[0]
    underperformer = regions_sorted[-1]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Top performer alert
        st.success(
            f"**🏆 Top Performer:** {top_performer[0]} "
            f"({top_performer[1]['performance']:.1f}% | ${top_performer[1]['revenue']:.1f}M)"
        )
    
    with col2:
        # Underperformer alert
        if underperformer[1]['performance'] < 100:
            st.error(
                f"**⚠️ Needs Attention:** {underperformer[0]} "
                f"({underperformer[1]['performance']:.1f}% | ${underperformer[1]['revenue']:.1f}M)"
            )
        else:
            st.warning(
                f"**📈 Room for Growth:** {underperformer[0]} "
                f"({underperformer[1]['performance']:.1f}% | ${underperformer[1]['revenue']:.1f}M)"
            )
    
    with col3:
        # Action buttons
        if st.button("🔍 Deep Dive Analysis", key="deep_dive", use_container_width=True):
            st.info("Launching detailed regional analysis...")
        
        if st.button("📊 Generate Report", key="generate_report", use_container_width=True):
            st.info("Generating executive summary report...")


def show_all_regions_analysis(regions):
    """Display analysis for all regions with comparison capabilities."""

    st.markdown("### 🌎 All Regions Comparative Analysis")

    # Regional comparison matrix
    col1, col2 = st.columns([3, 1])

    with col1:
        # Performance matrix heatmap
        st.markdown("#### Performance Matrix")

        metrics = [
            "Revenue",
            "Performance",
            "Growth Rate",
            "Efficiency",
            "Customer Sat",
        ]
        region_names = list(regions.keys())

        # Generate matrix data
        matrix_data = []
        for region in region_names:
            row = [
                regions[region]["revenue"],
                regions[region]["performance"],
                np.random.uniform(8, 16),  # Growth rate
                np.random.uniform(90, 110),  # Efficiency
                np.random.uniform(4.5, 4.9),  # Customer satisfaction
            ]
            matrix_data.append(row)

        matrix_df = pd.DataFrame(matrix_data, columns=metrics, index=region_names)

        # Normalize for heatmap
        matrix_normalized = matrix_df.div(matrix_df.max()) * 100

        fig_heatmap = px.imshow(
            matrix_normalized,
            title="Regional Performance Heatmap (Normalized %)",
            color_continuous_scale="RdYlGn",
            aspect="auto",
        )
        fig_heatmap.update_layout(height=300)
        st.plotly_chart(fig_heatmap, use_container_width=True)

    with col2:
        st.markdown("#### Quick Actions")

        actions = [
            ("📊 Deep Dive Analysis", "warning"),
            ("🎯 Set Regional Targets", "info"),
            ("📈 Launch Initiative", "success"),
            ("⚠️ Address Gaps", "error"),
        ]

        for action, color in actions:
            {
                "warning": "#ffc107",
                "info": "#17a2b8",
                "success": "#28a745",
                "error": "#dc3545",
            }[color]

            if st.button(action, key=f"regional_{action}", use_container_width=True):
                st.success(f"{action} initiated for selected regions!")


def show_single_region_analysis(region_name, region_data):
    """Display detailed analysis for a specific region."""

    st.markdown(f"### 🎯 {region_name} - Detailed Regional Analysis")

    # Region-specific KPIs
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-value-container">
                <div class="metric-value">${region_data['revenue']:.1f}M</div>
                <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">+8.3%</div>
            </div>
            <div class="metric-label">Regional Revenue</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-value-container">
                <div class="metric-value">{region_data['performance']:.1f}%</div>
                <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">+2.1%</div>
            </div>
            <div class="metric-label">Performance Score</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-value-container">
                <div class="metric-value">{region_data["stores"]}</div>
                <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">+3 stores</div>
            </div>
            <div class="metric-label">Store Count</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-value-container">
                <div class="metric-value">23.4%</div>
                <div class="metric-badge" style="background: #dcfce7; color: #16a34a;">+1.2%</div>
            </div>
            <div class="metric-label">Market Share</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Market breakdown within region
    st.markdown("#### 🏙️ Market Performance Within Region")

    markets_data = pd.DataFrame(
        {
            "Market": region_data["markets"],
            "Revenue": np.random.uniform(2, 5, len(region_data["markets"])),
            "Performance": np.random.uniform(95, 115, len(region_data["markets"])),
            "Store_Count": np.random.randint(8, 25, len(region_data["markets"])),
        }
    )

    col1, col2 = st.columns(2)

    with col1:
        fig_market_revenue = px.bar(
            markets_data,
            x="Market",
            y="Revenue",
            title=f"Market Revenue in {region_name} ($M)",
            color="Performance",
            color_continuous_scale="RdYlGn",
        )
        st.plotly_chart(fig_market_revenue, use_container_width=True)

    with col2:
        fig_market_performance = px.scatter(
            markets_data,
            x="Store_Count",
            y="Performance",
            size="Revenue",
            hover_name="Market",
            title="Performance vs Store Count by Market",
        )
        st.plotly_chart(fig_market_performance, use_container_width=True)


def show_market_analysis():
    """Display market-level analysis."""
    st.markdown("### 🏢 Market-Level Analysis")
    st.info(
        "🚧 Market analysis view - drill down to individual market performance, competitor analysis, and local trends."
    )


def show_district_analysis():
    """Display district-level analysis."""
    st.markdown("### 🏪 District-Level Analysis")
    st.info(
        "🚧 District analysis view - operational metrics, district manager performance, and store clustering analysis."
    )


def show_store_analysis():
    """Display individual store analysis."""
    st.markdown("### 🏬 Individual Store Analysis")
    st.info(
        "🚧 Store-level analysis view - individual store deep dive, performance benchmarking, and optimization recommendations."
    )
