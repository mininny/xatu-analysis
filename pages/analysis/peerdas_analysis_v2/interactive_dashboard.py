"""
Interactive dashboard for Head Correctness analysis.

Analyzes head correctness (voting for proposed block_roots, including those
that may have been reorged) with bucketing by blob count and filtering by 
proposer and attester characteristics.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone, time
from typing import Dict, Any, Optional, List
import logging
from urllib.parse import urlencode

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import shared components
from shared.header import render_global_header, get_global_cluster, get_global_network
from shared.database import get_database_connection
from shared.ethereum.validator_filters import (
    create_proposer_filters_ui,
    create_attester_filters_ui,
    get_node_classifications
)

# Import local modules
from pages.analysis.peerdas_analysis_v2.loader import (
    load_eligible_slots,
    load_head_correctness_data,
    validate_data_availability,
    get_unique_clients,
    load_network_mapping
)
from pages.analysis.peerdas_analysis_v2.plot_generators import (
    create_head_correctness_boxplot,
    create_head_correctness_chart,
    create_head_correctness_violin,
    create_head_correctness_ridgeline,
    create_advanced_grouped_boxplot,
    create_head_correctness_ecdf,
    create_head_correctness_cdf,
    create_head_correctness_summary,
    create_head_correctness_bar
)


def parse_url_params() -> Dict[str, Any]:
    """Parse URL query parameters and return configuration dict."""
    params = st.query_params
    config = {}

    # Parse period parameter (for quick time selections)
    if 'period' in params:
        config['period'] = params['period']

    # Parse datetime parameters
    if 'start_date' in params:
        try:
            config['start_datetime'] = datetime.fromisoformat(params['start_date'])
        except:
            pass

    if 'end_date' in params:
        try:
            config['end_datetime'] = datetime.fromisoformat(params['end_date'])
        except:
            pass

    # Parse time components for custom selection
    if 'start_time' in params:
        config['start_time'] = params['start_time']

    if 'end_time' in params:
        config['end_time'] = params['end_time']

    # Parse integer parameters
    if 'num_buckets' in params:
        try:
            config['num_buckets'] = int(params['num_buckets'])
        except:
            pass

    # Parse string parameters
    for key in ['grouping_dimension', 'attester_grouping', 'mev_filter', 'view_mode', 'chart_type',
                'proposer_type', 'attester_type', 'scatter_aggregation']:
        if key in params:
            config[key] = params[key]

    # Parse list parameters (comma-separated)
    for key in ['proposer_cl', 'proposer_el', 'attester_cl', 'attester_el']:
        if key in params:
            values = params[key].split(',')
            config[key] = [v.strip() for v in values if v.strip()]

    # Parse float parameters
    if 'performance_threshold' in params:
        try:
            config['performance_threshold'] = float(params['performance_threshold'])
        except:
            pass

    # Parse boolean parameters
    if 'filter_zero_blobs' in params:
        config['filter_zero_blobs'] = params['filter_zero_blobs'].lower() == 'true'
    if 'filter_low_data_buckets' in params:
        config['filter_low_data_buckets'] = params['filter_low_data_buckets'].lower() == 'true'

    return config


def generate_url_params(config: Dict[str, Any]) -> str:
    """Generate URL parameters from configuration."""
    params = {}

    # Add period parameter
    if 'period' in config and config['period']:
        params['period'] = config['period']

    # Add datetime parameters (use absolute timestamps)
    if 'start_datetime' in config and config['start_datetime']:
        params['start_date'] = config['start_datetime'].isoformat()

    if 'end_datetime' in config and config['end_datetime']:
        params['end_date'] = config['end_datetime'].isoformat()

    # Add time components for custom selection
    if 'start_time' in config and config['start_time']:
        params['start_time'] = config['start_time']

    if 'end_time' in config and config['end_time']:
        params['end_time'] = config['end_time']

    # Add other parameters
    simple_params = ['num_buckets', 'grouping_dimension', 'attester_grouping', 'mev_filter', 'view_mode',
                     'chart_type', 'proposer_type', 'attester_type',
                     'scatter_aggregation', 'performance_threshold', 'filter_zero_blobs', 'filter_low_data_buckets']

    for key in simple_params:
        if key in config and config[key] is not None:
            params[key] = str(config[key])

    # Add list parameters (comma-separated)
    list_params = ['proposer_cl', 'proposer_el', 'attester_cl', 'attester_el']
    for key in list_params:
        if key in config and config[key]:
            params[key] = ','.join(config[key])

    # Don't include load_data or show_trend_line in URL
    params.pop('load_data', None)
    params.pop('show_trend_line', None)

    return params


def update_url_with_config(config: Dict[str, Any]):
    """Update the browser URL with the current configuration."""
    params = generate_url_params(config)
    st.query_params.update(params)


def initialize_session_state():
    """Initialize session state variables."""
    if 'peerdas_v2_data_loaded' not in st.session_state:
        st.session_state.peerdas_v2_data_loaded = False
    if 'peerdas_v2_analysis_data' not in st.session_state:
        st.session_state.peerdas_v2_analysis_data = {}
    if 'peerdas_v2_last_config' not in st.session_state:
        st.session_state.peerdas_v2_last_config = None
    
    # Load URL parameters on first run
    if 'peerdas_v2_url_params_loaded' not in st.session_state:
        st.session_state.peerdas_v2_url_params_loaded = True
        st.session_state.peerdas_v2_url_config = parse_url_params()


def render_sidebar_config(cluster: str, network: str) -> Dict[str, Any]:
    """
    Render sidebar configuration options.
    
    Returns:
        Configuration dictionary
    """
    st.sidebar.header("⚙️ Configuration")
    
    # Check if network changed to clear cache
    if 'peerdas_v2_last_network' not in st.session_state:
        st.session_state.peerdas_v2_last_network = network
    elif st.session_state.peerdas_v2_last_network != network:
        st.session_state.peerdas_v2_last_network = network
        st.session_state.peerdas_v2_data_loaded = False
        st.session_state.peerdas_v2_analysis_data = {}
        logger.info(f"Network changed to {network}, clearing cache")
    
    # Check if cluster changed
    if 'peerdas_v2_last_cluster' not in st.session_state:
        st.session_state.peerdas_v2_last_cluster = cluster
    elif st.session_state.peerdas_v2_last_cluster != cluster:
        st.session_state.peerdas_v2_last_cluster = cluster
        st.session_state.peerdas_v2_data_loaded = False
        st.session_state.peerdas_v2_analysis_data = {}
        logger.info(f"Cluster changed to {cluster}, clearing cache")
    
    # Auto-select experimental cluster for fusaka networks
    if 'fusaka' in network.lower() and cluster != 'experimental':
        cluster = 'experimental'
        logger.info(f"Using experimental cluster for {network}")
        st.sidebar.info(f"Auto-selected experimental cluster for {network}")
    
    if 'sunnyside' in network.lower() and cluster != 'sunnyside':
        cluster = 'sunnyside'
        logger.info(f"Using sunnyside cluster for {network}")
        st.sidebar.info(f"Auto-selected sunnyside cluster for {network}")
    
    # Get URL parameters
    url_config = st.session_state.get('peerdas_v2_url_config', {})
    
    # Grouping selection (moved to top)
    st.sidebar.subheader("🧩 Grouping")
    
    # Proposer grouping
    proposer_options = ['none', 'node_type', 'cl_client', 'el_client', 'architecture', 'operator', 'region', 'datacenter', 'cl_el_combined', 'cl_node_type', 'cl_architecture', 'cl_operator', 'block_building', 'node_type_mev', 'cl_node_type_mev']
    proposer_grouping = st.sidebar.selectbox(
        "Proposer Grouping",
        options=proposer_options,
        index=proposer_options.index(url_config.get('grouping_dimension', 'none')),
        format_func=lambda x: {
            'none': 'None (All Proposers)',
            'node_type': 'Node Type',
            'cl_client': 'CL Client',
            'el_client': 'EL Client',
            'architecture': 'Architecture',
            'operator': 'Operator',
            'region': 'Region',
            'datacenter': 'Datacenter',
            'cl_el_combined': 'CL+EL Combination',
            'cl_node_type': 'CL+Node Type',
            'cl_architecture': 'CL+Architecture',
            'cl_operator': 'CL+Operator',
            'block_building': 'Block Building Method',
            'node_type_mev': 'Node Type + Block Building',
            'cl_node_type_mev': 'CL+Node Type + Block Building'
        }[x],
        help="Group slots by proposer characteristics"
    )

    # Attester grouping (new)
    attester_options = ['none', 'node_type', 'cl_client', 'el_client', 'architecture', 'operator', 'region', 'datacenter', 'cl_el_combined', 'cl_node_type', 'el_node_type', 'cl_architecture', 'cl_operator']
    attester_grouping = st.sidebar.selectbox(
        "Attester Grouping",
        options=attester_options,
        index=attester_options.index(url_config.get('attester_grouping', 'none')),
        format_func=lambda x: {
            'none': 'None (All Attesters)',
            'node_type': 'Node Type',
            'cl_client': 'CL Client',
            'el_client': 'EL Client',
            'architecture': 'Architecture',
            'operator': 'Operator',
            'region': 'Region',
            'datacenter': 'Datacenter',
            'cl_el_combined': 'CL+EL Combination',
            'cl_node_type': 'CL+Node Type',
            'el_node_type': 'EL+Node Type',
            'cl_architecture': 'CL+Architecture',
            'cl_operator': 'CL+Operator'
        }[x],
        help="Group attestations by attester characteristics"
    )
    
    # For backward compatibility, keep grouping_dimension as proposer_grouping
    grouping_dimension = proposer_grouping

    # Time range selection
    st.sidebar.subheader("📅 Time Range")

    # Quick period selection
    time_options = {
        "Last 1 Hour": timedelta(hours=1),
        "Last 2 Hours": timedelta(hours=2),
        "Last 6 Hours": timedelta(hours=6),
        "Last 12 Hours": timedelta(hours=12),
        "Last 24 Hours": timedelta(hours=24),
        "Last 3 Days": timedelta(days=3),
        "Last 7 Days": timedelta(days=7),
        "Custom": None
    }

    # Determine default period based on URL parameters
    saved_period = url_config.get('period', None)

    # Check if we should use Custom mode
    if (saved_period and saved_period in time_options) and not (
        'start_datetime' in url_config or 'end_datetime' in url_config
    ):
        # Use the saved period
        default_period_index = list(time_options.keys()).index(saved_period)
    elif 'start_datetime' in url_config or 'end_datetime' in url_config:
        # If specific dates are provided, use Custom
        default_period_index = list(time_options.keys()).index("Custom")
    else:
        # Default to "Last 2 Hours"
        default_period_index = list(time_options.keys()).index("Last 2 Hours")

    selected_period = st.sidebar.selectbox(
        "Quick Period Selection",
        options=list(time_options.keys()),
        index=default_period_index,
        key="peerdas_v2_period_selector",
        help="Select a predefined period or choose Custom for manual selection"
    )

    # Set dates based on selection
    if selected_period != "Custom":
        time_delta = time_options[selected_period]
        end_datetime = datetime.now(timezone.utc).replace(tzinfo=None)
        start_datetime = end_datetime - time_delta

        # Show the selected range (read-only)
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.date_input(
                "Start Date",
                value=start_datetime.date(),
                disabled=True,
                key="peerdas_v2_start_date_display"
            )
            st.time_input(
                "Start Time (UTC)",
                value=start_datetime.time(),
                disabled=True,
                key="peerdas_v2_start_time_display",
                step=300
            )
        with col2:
            st.date_input(
                "End Date",
                value=end_datetime.date(),
                disabled=True,
                key="peerdas_v2_end_date_display"
            )
            st.time_input(
                "End Time (UTC)",
                value=end_datetime.time(),
                disabled=True,
                key="peerdas_v2_end_time_display",
                step=300
            )
    else:
        # Custom date selection
        # Get defaults from URL or use last 2 hours
        default_end = url_config.get('end_datetime') or datetime.now(timezone.utc).replace(tzinfo=None)
        default_start = url_config.get('start_datetime') or (default_end - timedelta(hours=2))
        
        # Initialize session state for time widgets only once
        if "peerdas_v2_start_time" not in st.session_state:
            st.session_state["peerdas_v2_start_time"] = default_start.time()
        if "peerdas_v2_end_time" not in st.session_state:
            st.session_state["peerdas_v2_end_time"] = default_end.time()

        # Custom date and time inputs
        st.sidebar.subheader("Start Time")
        start_col1, start_col2 = st.sidebar.columns(2)
        start_date = start_col1.date_input(
            "Start Date",
            value=default_start.date(),
            max_value=datetime.now().date()
        )
        start_time = start_col2.time_input(
            "Start Time (UTC)",
            key="peerdas_v2_start_time",  # Use key to get/set value
            step=300  # 5 minute steps
        )

        st.sidebar.subheader("End Time")
        end_col1, end_col2 = st.sidebar.columns(2)
        end_date = end_col1.date_input(
            "End Date",
            value=default_end.date(),
            max_value=datetime.now().date()
        )
        end_time = end_col2.time_input(
            "End Time (UTC)",
            key="peerdas_v2_end_time",  # Use key to get/set value
            step=300  # 5 minute steps
        )

        start_datetime = datetime.combine(start_date, start_time)
        end_datetime = datetime.combine(end_date, end_time)
    
    # Bucketing options for blob count
    st.sidebar.subheader("🗂️ Blob Count Bucketing")

    # Add checkbox to filter out 0 blob slots (enabled by default)
    filter_zero_blobs = st.sidebar.checkbox(
        "Filter out 0 blob slots",
        value=url_config.get('filter_zero_blobs', True),
        help="Exclude slots with 0 blobs from the analysis"
    )

    # Add filter option for low-data buckets
    filter_low_data_buckets = st.sidebar.checkbox(
        "Filter out low-data buckets",
        value=url_config.get('filter_low_data_buckets', True),
        help="Hide buckets with insufficient data points (less than 2% of total slots). Uncheck to show all buckets even if they have no data."
    )

    num_buckets = st.sidebar.slider(
        "Number of Buckets",
        min_value=1,
        max_value=12,
        value=url_config.get('num_buckets', 10),
        help="Number of buckets to divide blob counts into (automatically calculates bucket size based on data range)"
    )

    # MEV filtering
    st.sidebar.subheader("🎰 Block Building Filter")
    mev_filter = st.sidebar.selectbox(
        "Block Source",
        options=['both', 'yes', 'no'],
        index=['both', 'yes', 'no'].index(url_config.get('mev_filter', 'both')),
        format_func=lambda x: {
            'both': 'All Blocks',
            'yes': 'MEV Relay Blocks Only',
            'no': 'Locally Built Blocks Only'
        }[x],
        help="Filter slots based on whether blocks were delivered via MEV relay or built locally"
    )
    
    # Create filter UI components using shared utility with URL parameters
    with st.sidebar:
        # Prepare initial values from URL config
        proposer_initial = {}
        attester_initial = {}
        
        if url_config:
            # Convert URL params to the format expected by the filter functions
            if 'proposer_type' in url_config:
                proposer_initial['proposer_type'] = url_config['proposer_type']
            if 'proposer_cl' in url_config:
                proposer_initial['proposer_cl'] = url_config['proposer_cl']
            if 'proposer_el' in url_config:
                proposer_initial['proposer_el'] = url_config['proposer_el']
                
            if 'attester_type' in url_config:
                attester_initial['attester_type'] = url_config['attester_type']
            if 'attester_cl' in url_config:
                attester_initial['attester_cl'] = url_config['attester_cl']
            if 'attester_el' in url_config:
                attester_initial['attester_el'] = url_config['attester_el']
        
        proposer_filters = create_proposer_filters_ui(network, cluster_name=cluster, initial_values=proposer_initial)
        attester_filters = create_attester_filters_ui(network, cluster_name=cluster, initial_values=attester_initial)
    
    # Load data button
    st.sidebar.markdown("---")
    
    # Chart Options
    st.sidebar.subheader("📊 Chart Options")
    
    # Add view mode selector
    view_mode = st.sidebar.radio(
        "View Mode",
        options=['correct', 'incorrect'],
        index=['correct', 'incorrect'].index(url_config.get('view_mode', 'correct')),
        format_func=lambda x: {
            'correct': '✅ Correct Votes',
            'incorrect': '❌ Incorrect/Missing Votes'
        }[x],
        help="Show head correctness or incorrectness (inverse)"
    )
    
    chart_type = st.sidebar.selectbox(
        "Chart Type",
        options=['boxplot', 'violin', 'ridgeline', 'scatter', 'bar', 'ecdf_diff', 'cdf', 'summary'],
        index=['boxplot', 'violin', 'ridgeline', 'scatter', 'bar', 'ecdf_diff', 'cdf', 'summary'].index(
            url_config.get('chart_type', 'boxplot')
        ),
        format_func=lambda x: {
            'boxplot': 'Box Plot Distribution',
            'violin': 'Violin Plot Distribution',
            'ridgeline': 'Ridgeline Plot (Joy Plot)',
            'scatter': 'Scatter Plot with Trend',
            'bar': 'Bar Chart Comparison',
            'ecdf_diff': 'Difference ECDF',
            'cdf': 'Cumulative Distribution (CDF)',
            'summary': 'Statistical Summary Table'
        }[x]
    )
    
    show_trend_line = False
    scatter_aggregation = url_config.get('scatter_aggregation', 'p95')
    performance_threshold = url_config.get('performance_threshold', 95.0)
    
    if chart_type == 'summary':
        performance_threshold = st.sidebar.slider(
            "Performance Threshold (%)",
            min_value=50.0,
            max_value=100.0,
            value=url_config.get('performance_threshold', 95.0),
            step=0.5,
            help="Threshold for considering a slot as 'good' performance. Slots with head correctness ≥ this value are counted as meeting the threshold."
        )
    elif chart_type == 'scatter':
        show_trend_line = st.sidebar.checkbox(
            "Show trend line",
            value=True,
            help="Display trend line on scatter plot"
        )
        
        agg_options = ['mean', 'median', 'p25', 'p50', 'p75', 'p90', 'p95', 'p99', 'min', 'max']
        default_agg = url_config.get('scatter_aggregation', 'p95')
        if default_agg not in agg_options:
            default_agg = 'p95'
        
        scatter_aggregation = st.sidebar.selectbox(
            "Aggregation Method",
            options=agg_options,
            index=agg_options.index(default_agg),
            format_func=lambda x: {
                'mean': 'Mean (Average)',
                'median': 'Median',
                'p25': '25th Percentile',
                'p50': '50th Percentile (Median)',
                'p75': '75th Percentile',
                'p90': '90th Percentile',
                'p95': '95th Percentile',
                'p99': '99th Percentile',
                'min': 'Minimum',
                'max': 'Maximum'
            }[x],
            help="Choose how to aggregate head correctness values for each blob count bucket"
        )
    
    st.sidebar.markdown("---")
    
    col1, col2 = st.sidebar.columns([2, 1])
    with col1:
        load_data = st.sidebar.button("🚀 Load Data", type="primary", use_container_width=True)
    with col2:
        if st.sidebar.button("🗑️ Clear", use_container_width=True):
            st.cache_data.clear()
            st.session_state.peerdas_v2_analysis_data = {}
            st.session_state.peerdas_v2_data_loaded = False
            # Force clear all caches
            st.rerun()
            st.sidebar.success("Cache cleared and page refreshed!")
    
    # Combine all configuration
    config = {
        'cluster': cluster,
        'network': network,
        'start_datetime': start_datetime,
        'end_datetime': end_datetime,
        'num_buckets': num_buckets,
        'filter_zero_blobs': filter_zero_blobs,  # Add filter for zero blob slots
        'grouping_dimension': grouping_dimension,
        'attester_grouping': attester_grouping,  # Add attester grouping
        'mev_filter': mev_filter,
        'view_mode': view_mode,
        'chart_type': chart_type,
        'show_trend_line': show_trend_line,
        'scatter_aggregation': scatter_aggregation,
        'performance_threshold': performance_threshold,
        'filter_low_data_buckets': filter_low_data_buckets,  # Add filter for low-data buckets
        'load_data': load_data
    }
    

    # Add period if not custom
    if selected_period != "Custom":
        config['period'] = selected_period
    
    # Add filter values from shared utility
    config.update(proposer_filters)
    config.update(attester_filters)
    
    return config


def load_and_process_head_correctness_data(config: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    Load and process head correctness data.

    Args:
        config: Configuration dictionary

    Returns:
        Processed DataFrame with head correctness data or None if loading fails
    """
    with st.spinner("🔄 Loading head correctness data..."):
        try:
            # Load eligible slots (filtered by proposer and MEV status)
            try:
                eligible_slots, slot_to_block, slot_to_proposer, mev_slots = load_eligible_slots(
                    network=config['network'],
                    start_date=config['start_datetime'],
                    end_date=config['end_datetime'],
                    proposer_type=config.get('proposer_type'),
                    cl_filter=config.get('proposer_cl'),
                    el_filter=config.get('proposer_el'),
                    architecture_filter=config.get('proposer_architecture'),
                    operator_filter=config.get('proposer_operator'),
                    region_filter=config.get('proposer_region'),
                    datacenter_filter=config.get('proposer_datacenter'),
                    mev_filter=config.get('mev_filter'),
                    cluster_name=config['cluster']
                )
            except Exception as e:
                # Display the actual backend error
                st.error(f"🚨 Backend Error: {str(e)}")

                # Check if there's additional error context in session state
                if 'peerdas_v2_last_error' in st.session_state:
                    with st.expander("🔍 Error Details", expanded=True):
                        st.code(st.session_state['peerdas_v2_last_error'])

                # Clear cache to allow retry
                st.cache_data.clear()
                return None

            if not eligible_slots:
                st.error("No eligible slots found for the selected proposer filters and time range")
                
                # Show more detailed debug info
                with st.expander("🐛 Debug: Why no slots?", expanded=False):
                    st.write("Possible reasons:")
                    st.write("1. Network spec might not be loading correctly")
                    st.write("2. Validator indices might not match any proposers in the time range")
                    st.write("3. The filters might be too restrictive")
                    st.write("4. There might be no blocks in the selected time range")
                    
                    # Try to query directly
                    try:
                        from shared.database import get_database_connection
                        conn = get_database_connection(config['cluster'])
                        if conn:
                            import pandas as pd
                            test_query = """
                            SELECT COUNT(*) as count, 
                                   MIN(slot_start_date_time) as min_time,
                                   MAX(slot_start_date_time) as max_time
                            FROM beacon_api_eth_v2_beacon_block
                            WHERE meta_network_name = %(network)s
                              AND slot_start_date_time BETWEEN %(start_date)s AND %(end_date)s
                            """
                            test_df = pd.read_sql(test_query, conn, params={
                                'network': config['network'],
                                'start_date': config['start_datetime'],
                                'end_date': config['end_datetime']
                            })
                            st.write("**Direct query results:**")
                            st.write(test_df)
                    except Exception as e:
                        st.write(f"Error running test query: {e}")
                        
                st.cache_data.clear()
                return None
            
            # Load head correctness data (against proposed blocks, including reorged)
            try:
                data = load_head_correctness_data(
                    network=config['network'],
                    start_date=config['start_datetime'],
                    end_date=config['end_datetime'],
                    eligible_slots=eligible_slots,
                    slot_to_block=slot_to_block,
                    slot_to_proposer=slot_to_proposer,
                    mev_slots=mev_slots,
                    mev_filter=config.get('mev_filter'),
                    proposer_type=config.get('proposer_type'),
                    proposer_cl_filter=config.get('proposer_cl'),
                    proposer_el_filter=config.get('proposer_el'),
                    proposer_architecture_filter=config.get('proposer_architecture'),
                    proposer_operator_filter=config.get('proposer_operator'),
                    proposer_region_filter=config.get('proposer_region'),
                    proposer_datacenter_filter=config.get('proposer_datacenter'),
                    attester_type=config.get('attester_type'),
                    cl_filter=config.get('attester_cl'),
                    el_filter=config.get('attester_el'),
                    architecture_filter=config.get('attester_architecture'),
                    operator_filter=config.get('attester_operator'),
                    region_filter=config.get('attester_region'),
                    datacenter_filter=config.get('attester_datacenter'),
                    grouping_dimension=config.get('grouping_dimension'),
                    attester_grouping_dimension=config.get('attester_grouping'),  # Pass attester grouping
                    cluster_name=config['cluster']
                )
            except Exception as e:
                # Display the actual backend error
                st.error(f"🚨 Backend Error Loading Head Correctness: {str(e)}")

                # Check if there's additional error context in session state
                if 'peerdas_v2_last_error' in st.session_state:
                    with st.expander("🔍 Error Details", expanded=True):
                        st.code(st.session_state['peerdas_v2_last_error'])

                # Clear cache to allow retry
                st.cache_data.clear()
                st.session_state.peerdas_v2_analysis_data = {}
                st.session_state.peerdas_v2_data_loaded = False
                return None

            if data.empty:
                st.error("No head correctness data returned!")
                # Check if there were any SQL errors in the session state
                if hasattr(st, '_last_sql_error'):
                    st.error(f"SQL Error: {st._last_sql_error}")
                    delattr(st, '_last_sql_error')
                else:
                    st.warning("""
                    No head correctness data found for the selected filters.

                    **Possible causes:**
                    - No data_column_sidecar data available for the selected time range
                    - No committee data available
                    - No attestation data found for the eligible slots

                    **Note:** PeerDAS analysis requires data_column_sidecar data to determine blob counts.
                    Try selecting a different time range where data_column_sidecar data is available.
                    """)
                # Clear cache to avoid bad data persistence
                st.cache_data.clear()
                st.session_state.peerdas_v2_analysis_data = {}
                st.session_state.peerdas_v2_data_loaded = False
                return None

            # Apply filter for zero blob slots if enabled
            if config.get('filter_zero_blobs', True) and 'blob_count' in data.columns:
                original_count = len(data)
                data = data[data['blob_count'] > 0].copy()
                logger.info(f"Filtered out {original_count - len(data)} rows with 0 blob count")

                if data.empty:
                    st.warning("All slots had 0 blobs. Try disabling the '**Filter out 0 blob slots**' option to see data.")
                    return None

            # Compute slot-level coverage after filtering to only slots with committee data
            slots_in_result = data['slot'].nunique() if 'slot' in data.columns else 0
            eligible_count = len(eligible_slots)
            filtered_out = max(eligible_count - slots_in_result, 0)

            # Store in session state
            st.session_state.peerdas_v2_analysis_data = {
                'raw_data': data,
                'unique_slots': eligible_count,
                'total_slots_analyzed': slots_in_result,
                'filtered_out_slots': filtered_out,
                'avg_head_correctness': data['head_correctness_pct'].mean() if 'head_correctness_pct' in data.columns else 0,
                'mev_slots': mev_slots if mev_slots else [],
                'filter_low_data_buckets': config.get('filter_low_data_buckets', True)  # Store filter setting
            }
            logger.info(f"Stored {len(mev_slots) if mev_slots else 0} MEV slots in session state")
            st.session_state.peerdas_v2_data_loaded = True
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading attestation data: {e}")
            st.error(f"🚨 Failed to load data: {str(e)}")

            # Check if there's additional error context in session state
            if 'peerdas_v2_last_error' in st.session_state:
                with st.expander("🔍 Full Error Details", expanded=True):
                    st.code(st.session_state['peerdas_v2_last_error'])

            # Clear cache to allow retry
            st.cache_data.clear()
            st.session_state.peerdas_v2_analysis_data = {}
            st.session_state.peerdas_v2_data_loaded = False
            return None


def main():
    """Main dashboard function."""
    initialize_session_state()
    
    # Render header and get cluster/network
    render_global_header()
    cluster = get_global_cluster()
    network = get_global_network()
    # Render sidebar configuration
    config = render_sidebar_config(cluster, network)
    
    # Load and display data if requested
    if config['load_data']:
        # Update URL with current configuration
        update_url_with_config(config)
        
        data = load_and_process_head_correctness_data(config)
        
        if data is not None and not data.empty:
            # Transform data for incorrect view mode if selected
            if config.get('view_mode') == 'incorrect':
                data = data.copy()
                data['head_correctness_pct'] = 100 - data['head_correctness_pct']
                view_mode_label = "Incorrectness"
            else:
                view_mode_label = "Correctness"
            
            # Create the visualization
            st.markdown("---")
            
            # Show shareable link
            with st.expander("🔗 Share This Configuration", expanded=False):
                st.info("The URL has been updated with your current configuration. Copy the URL from your browser's address bar to share this exact view.")
                st.code(f"Configuration loaded at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
            
            # Pre-calculate unique slot counts per bucket from combined data
            # This ensures consistent counts between proposer and attester charts
            # Use only proposer data (or the first data type) to avoid double-counting slots
            bucket_slot_counts = {}
            if 'blob_count' in data.columns and 'slot' in data.columns:
                # If we have data_type column, use only one type to avoid double-counting
                count_data = data
                if 'data_type' in data.columns:
                    # Prefer proposer data if available, otherwise use first type
                    if 'proposer' in data['data_type'].values:
                        count_data = data[data['data_type'] == 'proposer'].copy()
                    else:
                        first_type = data['data_type'].iloc[0]
                        count_data = data[data['data_type'] == first_type].copy()

                if config.get('num_buckets') == 1:
                    # Single bucket
                    bucket_slot_counts['All'] = count_data['slot'].nunique()
                elif config.get('num_buckets') and config.get('num_buckets') > 1:
                    # Multiple buckets - calculate for each
                    max_blobs = int(count_data['blob_count'].max())
                    min_blobs = int(count_data['blob_count'].min())
                    blob_range = max_blobs - min_blobs + 1
                    num_buckets = config.get('num_buckets')

                    # If we can't divide evenly, reduce number of buckets
                    if blob_range < num_buckets:
                        num_buckets = blob_range
                    elif blob_range % num_buckets != 0:
                        # Find the largest number of buckets that works well
                        for actual_buckets in range(num_buckets, 0, -1):
                            if blob_range % actual_buckets == 0 or actual_buckets == 1:
                                num_buckets = actual_buckets
                                break
                            bucket_size = blob_range // actual_buckets + (1 if blob_range % actual_buckets > 0 else 0)
                            if bucket_size * actual_buckets >= blob_range:
                                num_buckets = actual_buckets
                                break

                    # Create non-overlapping bucket edges
                    edges = []
                    current = min_blobs
                    bucket_size = blob_range // num_buckets
                    remainder = blob_range % num_buckets

                    for i in range(num_buckets):
                        edges.append(current)
                        # Distribute remainder across first buckets
                        current += bucket_size + (1 if i < remainder else 0)
                    edges.append(max_blobs + 1)  # Final edge

                    data_temp = count_data.copy()
                    data_temp['blob_bucket'] = pd.cut(data_temp['blob_count'], bins=edges, include_lowest=True, right=False)
                    data_temp['blob_bucket_label'] = data_temp['blob_bucket'].apply(
                        lambda x: f"{int(x.left)}-{int(x.right-1)}" if pd.notna(x) else "Unknown"
                    )
                    for bucket_label in data_temp['blob_bucket_label'].unique():
                        if pd.notna(bucket_label):
                            bucket_data = data_temp[data_temp['blob_bucket_label'] == bucket_label]
                            bucket_slot_counts[bucket_label] = bucket_data['slot'].nunique()
                else:
                    # No bucketing - count by individual blob counts
                    for blob_count in sorted(count_data['blob_count'].dropna().unique()):
                        bucket_data = count_data[count_data['blob_count'] == blob_count]
                        bucket_slot_counts[blob_count] = bucket_data['slot'].nunique()

            # Prepare metadata
            # Calculate slot counts based on what's actually being displayed in the chart
            total_slots = st.session_state.peerdas_v2_analysis_data.get('unique_slots', 0)
            filter_low_data_buckets = config.get('filter_low_data_buckets', True)
            
            # Calculate slots that will be displayed based on bucket filtering
            # Recalculate bucket slot counts using the same logic as chart functions to ensure consistency
            if 'slot' in data.columns:
                # Calculate total unique slots in the actual data (for threshold calculation)
                actual_total_slots = data['slot'].nunique()
                
                # Recalculate bucket slot counts using the same logic as chart functions
                chart_bucket_slot_counts = {}
                if config.get('num_buckets', 10) > 0:
                    # Bucketed data - need to recreate the same bucketing logic as chart functions
                    if 'blob_count' in data.columns:
                        min_blobs = data['blob_count'].min()
                        max_blobs = data['blob_count'].max()
                        num_buckets = config.get('num_buckets', 10)
                        
                        if min_blobs == max_blobs:
                            # Single blob count, no bucketing needed
                            chart_bucket_slot_counts[str(int(min_blobs))] = data['slot'].nunique()
                        else:
                            # Create buckets using the same logic as bar chart function
                            bucket_width = (max_blobs - min_blobs + 1) / num_buckets
                            edges = [min_blobs + i * bucket_width for i in range(num_buckets + 1)]
                            edges[-1] = max_blobs + 1  # Ensure last edge includes max value

                            data_temp = data.copy()
                            data_temp['blob_bucket'] = pd.cut(data_temp['blob_count'], bins=edges, include_lowest=True, right=False)
                            data_temp['blob_bucket_label'] = data_temp['blob_bucket'].apply(
                                lambda x: f"{int(np.floor(x.left))}-{int(np.ceil(x.right)-1)}" if pd.notna(x) else "Unknown"
                            )
                            
                            for bucket_label in data_temp['blob_bucket_label'].unique():
                                if pd.notna(bucket_label):
                                    bucket_data = data_temp[data_temp['blob_bucket_label'] == bucket_label]
                                    chart_bucket_slot_counts[bucket_label] = bucket_data['slot'].nunique()
                else:
                    # Non-bucketed data - count by individual blob counts
                    for blob_count in sorted(data['blob_count'].dropna().unique()):
                        bucket_data = data[data['blob_count'] == blob_count]
                        chart_bucket_slot_counts[blob_count] = bucket_data['slot'].nunique()
                
                total_bucket_slots = sum(chart_bucket_slot_counts.values())
                
                if filter_low_data_buckets:
                    # When filtering is enabled, only count slots from buckets that meet the threshold
                    # This matches the logic in the chart generation functions
                    total_buckets = len(chart_bucket_slot_counts)
                    if total_buckets <= 1:
                        slot_threshold = 0
                    else:
                        # Use actual_total_slots for threshold calculation (same as chart functions)
                        base_percentage = 0.02
                        scaled_percentage = base_percentage * min(1.0, 6.0 / total_buckets)
                        threshold_percentage = max(0.005, scaled_percentage)
                        slot_threshold = max(1, actual_total_slots * threshold_percentage)
                    
                    # Count slots only from buckets that meet the threshold
                    total_slots_analyzed = sum(
                        count for count in chart_bucket_slot_counts.values() 
                        if count >= slot_threshold
                    )
                    filtered_out_slots = sum(
                        count for count in chart_bucket_slot_counts.values() 
                        if count < slot_threshold
                    )
                else:
                    # When filtering is disabled, count all slots from all buckets
                    total_slots_analyzed = total_bucket_slots
                    filtered_out_slots = 0
            else:
                # Fallback to original values if no slot data
                total_slots_analyzed = st.session_state.peerdas_v2_analysis_data.get('total_slots_analyzed', 0)
                filtered_out_slots = st.session_state.peerdas_v2_analysis_data.get('filtered_out_slots', 0)
            
            metadata = {
                'total_slots': total_slots,
                'total_slots_analyzed': total_slots_analyzed,
                'filtered_out_slots': filtered_out_slots,
                'view_mode': config.get('view_mode', 'correct'),
                'view_mode_label': view_mode_label,
                'unique_slots_in_data': data['slot'].nunique() if 'slot' in data.columns else 0,  # Track actual unique slots
                'bucket_slot_counts': bucket_slot_counts,  # Pre-calculated slot counts per bucket
                'filter_low_data_buckets': filter_low_data_buckets  # Include filter setting in metadata
            }
            
            time_range = f"{config['start_datetime'].strftime('%Y-%m-%d %H:%M')} to {config['end_datetime'].strftime('%Y-%m-%d %H:%M')}"
            
            # Prepare filter information for chart annotations
            proposer_filters = {
                'node_type': config.get('proposer_type'),
                'cl_filter': config.get('proposer_cl'),
                'el_filter': config.get('proposer_el')
            }
            
            attester_filters = {
                'node_type': config.get('attester_type'),
                'cl_filter': config.get('attester_cl'),
                'el_filter': config.get('attester_el')
            }
            
            # Create chart based on selected type
            # Check if we have both proposer and attester data
            has_proposer_data = 'data_type' in data.columns and 'proposer' in data['data_type'].values
            has_attester_data = 'data_type' in data.columns and 'attester' in data['data_type'].values
            
            # Helper function to create chart with correct grouping dimension
            def create_chart_for_data_type(data_subset, data_type_label, grouping_dim):
                """Create chart for a specific data type with appropriate grouping."""
                if config['chart_type'] == 'boxplot':
                    return create_head_correctness_boxplot(
                        data=data_subset,
                        num_buckets=config.get('num_buckets'),
                        network=config['network'],
                        time_range=time_range,
                        metadata=metadata,
                        grouping_dimension=grouping_dim,
                        proposer_filters=proposer_filters,
                        attester_filters=attester_filters,
                        filter_low_data_buckets=config.get('filter_low_data_buckets', True),
                        title_suffix=f"({data_type_label.capitalize()} Grouping)"
                    )
                elif config['chart_type'] == 'violin':
                    return create_head_correctness_violin(
                        data=data_subset,
                        num_buckets=config.get('num_buckets'),
                        network=config['network'],
                        time_range=time_range,
                        metadata=metadata,
                        grouping_dimension=grouping_dim,
                        proposer_filters=proposer_filters,
                        attester_filters=attester_filters,
                        filter_low_data_buckets=config.get('filter_low_data_buckets', True),
                        title_suffix=f"({data_type_label.capitalize()} Grouping)"
                    )
                elif config['chart_type'] == 'ridgeline':
                    return create_head_correctness_ridgeline(
                        data=data_subset,
                        num_buckets=config.get('num_buckets'),
                        network=config['network'],
                        time_range=time_range,
                        metadata=metadata,
                        grouping_dimension=grouping_dim,
                        proposer_filters=proposer_filters,
                        attester_filters=attester_filters,
                        filter_low_data_buckets=config.get('filter_low_data_buckets', True),
                        title_suffix=f"({data_type_label.capitalize()} Grouping)"
                    )
                elif config['chart_type'] == 'ecdf_diff':
                    return create_head_correctness_ecdf(
                        data=data_subset,
                        num_buckets=config.get('num_buckets'),
                        network=config['network'],
                        time_range=time_range,
                        metadata=metadata,
                        grouping_dimension=grouping_dim,
                        proposer_filters=proposer_filters,
                        attester_filters=attester_filters,
                        difference_mode=True,
                        filter_low_data_buckets=config.get('filter_low_data_buckets', True),
                        title_suffix=f"({data_type_label.capitalize()} Grouping)"
                    )
                elif config['chart_type'] == 'cdf':
                    return create_head_correctness_cdf(
                        data=data_subset,
                        num_buckets=config.get('num_buckets'),
                        network=config['network'],
                        time_range=time_range,
                        metadata=metadata,
                        grouping_dimension=grouping_dim,
                        proposer_filters=proposer_filters,
                        attester_filters=attester_filters,
                        filter_low_data_buckets=config.get('filter_low_data_buckets', True),
                        title_suffix=f"({data_type_label.capitalize()} Grouping)"
                    )
                elif config['chart_type'] == 'bar':
                    return create_head_correctness_bar(
                        data=data_subset,
                        num_buckets=config.get('num_buckets'),
                        network=config['network'],
                        time_range=time_range,
                        metadata=metadata,
                        grouping_dimension=grouping_dim,
                        proposer_filters=proposer_filters,
                        attester_filters=attester_filters,
                        filter_low_data_buckets=config.get('filter_low_data_buckets', True),
                        title_suffix=f"({data_type_label.capitalize()} Grouping)"
                    )
                else:  # scatter/line chart
                    return create_head_correctness_chart(
                        data=data_subset,
                        num_buckets=config.get('num_buckets') or 6,
                        network=config['network'],
                        time_range=time_range,
                        metadata=metadata,
                        show_trend_line=config.get('show_trend_line', True),
                        aggregation_method=config.get('scatter_aggregation', 'p95'),
                        grouping_dimension=grouping_dim,
                        proposer_filters=proposer_filters,
                        attester_filters=attester_filters,
                        filter_low_data_buckets=config.get('filter_low_data_buckets', True),
                        title_suffix=f"({data_type_label.capitalize()} Grouping)"
                    )
            
            # Create and display charts
            if has_proposer_data and has_attester_data:
                # We have both - create separate charts
                proposer_data = data[data['data_type'] == 'proposer'].copy()
                attester_data = data[data['data_type'] == 'attester'].copy()

                # Debug: Log unique slot counts
                logger.info(f"Proposer data: {len(proposer_data)} rows, {proposer_data['slot'].nunique()} unique slots")
                logger.info(f"Attester data: {len(attester_data)} rows, {attester_data['slot'].nunique()} unique slots")
                
                # Display proposer chart first
                proposer_fig = create_chart_for_data_type(
                    proposer_data, 
                    'proposer',
                    config.get('grouping_dimension') or 'node_type'
                )
                st.plotly_chart(proposer_fig, use_container_width=True)
                
                # Display attester chart (always show it, even with 'none' grouping)
                attester_fig = create_chart_for_data_type(
                    attester_data,
                    'attester',
                    config.get('attester_grouping', 'none')  # Default to 'none' if not specified
                )
                st.plotly_chart(attester_fig, use_container_width=True)
                
            elif has_proposer_data:
                # Only proposer data
                proposer_data = data[data['data_type'] == 'proposer'].copy()
                proposer_fig = create_chart_for_data_type(
                    proposer_data,
                    'proposer',
                    config.get('grouping_dimension') or 'node_type'
                )
                st.plotly_chart(proposer_fig, use_container_width=True)
                
            elif has_attester_data:
                # Only attester data
                attester_data = data[data['data_type'] == 'attester'].copy()
                attester_fig = create_chart_for_data_type(
                    attester_data,
                    'attester',
                    config.get('attester_grouping') or 'node_type'
                )
                st.plotly_chart(attester_fig, use_container_width=True)
                
            else:
                # No data type column - treat as proposer data for backward compatibility
                if config['chart_type'] == 'summary':
                    # Summary table doesn't need special handling
                    fig = create_head_correctness_summary(
                        data=data,
                        num_buckets=config.get('num_buckets'),
                        network=config['network'],
                        time_range=time_range,
                        metadata=metadata,
                        grouping_dimension=config.get('grouping_dimension') or 'node_type',
                        proposer_filters=proposer_filters,
                        attester_filters=attester_filters,
                        performance_threshold=config.get('performance_threshold', 95.0),
                        filter_low_data_buckets=config.get('filter_low_data_buckets', True)
                    )
                else:
                    fig = create_chart_for_data_type(
                        data,
                        'proposer',
                        config.get('grouping_dimension') or 'node_type'
                    )
                st.plotly_chart(fig, use_container_width=True)
            
            # Show debug information
            with st.expander("🔍 Debug Information", expanded=False):
                st.write("### Filter Settings")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Proposer Filters:**")
                    st.write(f"- Node Type: {config.get('proposer_type', 'All')}")
                    st.write(f"- CL Clients: {config.get('proposer_cl', 'All')}")
                    st.write(f"- EL Clients: {config.get('proposer_el', 'All')}")
                    st.write(f"- MEV Filter: {config.get('mev_filter', 'both')}")
                with col2:
                    st.write("**Attester Filters:**")
                    st.write(f"- Node Type: {config.get('attester_type', 'All')}")
                    st.write(f"- CL Clients: {config.get('attester_cl', 'All')}")
                    st.write(f"- EL Clients: {config.get('attester_el', 'All')}")
                
                st.write(f"**Time Range:** {config['start_datetime']} to {config['end_datetime']}")
                st.write(f"**Network:** {config['network']}")
                st.write(f"**Cluster:** {config['cluster']}")
                
                st.write("### Data Loading Results")
                if 'peerdas_v2_analysis_data' in st.session_state:
                    analysis_data = st.session_state.peerdas_v2_analysis_data
                    st.write(f"**Eligible slots returned:** {analysis_data.get('unique_slots', 0)}")
                    st.write(f"**MEV slots found:** {len(analysis_data.get('mev_slots', []))}")
                    st.write(f"**Slots analyzed:** {analysis_data.get('total_slots_analyzed', 0)}")
                    st.write(f"**Slots filtered out:** {analysis_data.get('filtered_out_slots', 0)}")
                    
                    if 'raw_data' in analysis_data and not analysis_data['raw_data'].empty:
                        data = analysis_data['raw_data']
                        if 'slot' in data.columns:
                            st.write(f"**Sample slots (first 10):** {sorted(data['slot'].unique())[:10]}")
                            st.write(f"**Slot range:** {data['slot'].min()} to {data['slot'].max()}")
                else:
                    st.write("No data loaded yet.")
            
            # Show data summary
            with st.expander("📊 Data Summary", expanded=False):
                # First row of metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Eligible Slots",
                        f"{metadata['total_slots']:,}",
                        help="Total slots that match the proposer filters"
                    )
                
                with col2:
                    st.metric(
                        "Analyzed Slots",
                        f"{metadata['total_slots_analyzed']:,}",
                        help="Slots with head correctness data calculated"
                    )
                
                with col3:
                    avg_correctness = st.session_state.peerdas_v2_analysis_data.get('avg_head_correctness', 0)
                    st.metric(
                        "Avg Head Correctness",
                        f"{avg_correctness:.1f}%",
                        help="Mean head correctness across all analyzed slots"
                    )
                
                with col4:
                    if 'blob_count' in data.columns:
                        max_blobs = data['blob_count'].max()
                        min_blobs = data['blob_count'].min()
                        st.metric(
                            "Blob Count Range",
                            f"{min_blobs}-{max_blobs}",
                            help=f"Range of blob counts in the analyzed data"
                        )
                
                # Second row with MEV info
                mev_slots = st.session_state.peerdas_v2_analysis_data.get('mev_slots', [])
                # Debug: Always show MEV info, even if empty
                col1, col2, col3, col4 = st.columns(4)
                mev_slots_set = set(mev_slots) if mev_slots else set()
                slots_in_data = set(data['slot'].unique()) if 'slot' in data.columns else set()
                mev_in_data = len(slots_in_data.intersection(mev_slots_set))
                non_mev_in_data = len(slots_in_data) - mev_in_data
                
                # Debug info
                with col1:
                    st.metric(
                        "MEV Relay Blocks",
                        f"{mev_in_data:,}",
                        help="Blocks delivered via MEV relay"
                    )
                
                with col2:
                    st.metric(
                        "Locally Built Blocks",
                        f"{non_mev_in_data:,}",
                        help="Blocks built locally by validators"
                    )
                
                with col3:
                    if len(slots_in_data) > 0:
                        mev_pct = (mev_in_data / len(slots_in_data)) * 100
                        st.metric(
                            "MEV Relay %",
                            f"{mev_pct:.1f}%",
                            help="Percentage of blocks from MEV relay"
                        )
                
                with col4:
                    st.metric(
                        "Total MEV Slots Found",
                        f"{len(mev_slots_set):,}",
                        help=f"Total MEV relay slots in the time range (some may be filtered out)"
                    )
                
                # If some slots were filtered out due to missing committee data, show a notice
                if metadata.get('filtered_out_slots', 0) > 0:
                    st.info(f"Filtered out {metadata['filtered_out_slots']:,} slot(s) with no committee data.")

                # Show info about zero blob filter
                if config.get('filter_zero_blobs', True):
                    st.info("🎯 Zero blob slots are excluded from this analysis (configurable in sidebar)")

                # Show grouping information if applicable
                if config.get('grouping_dimension'):
                    group_labels = {
                        'none': 'None (All Proposers)',
                        'node_type': 'Node Type',
                        'cl_client': 'CL Client',
                        'el_client': 'EL Client',
                        'cl_el_combined': 'CL+EL Combination',
                        'cl_node_type': 'CL+Node Type',
                        'block_building': 'Block Building Method',
                        'node_type_mev': 'Node Type + Block Building',
                        'cl_node_type_mev': 'CL+Node Type + Block Building'
                    }
                    grouping_info = f"Grouping by: {group_labels.get(config['grouping_dimension'], config['grouping_dimension'])}"
                    if config['chart_type'] == 'scatter':
                        grouping_info += " (showing p95 percentile)"
                    st.info(grouping_info)
                
                # Show MEV filter information
                if config.get('mev_filter'):
                    mev_info = {
                        'both': "Showing all blocks (MEV relay + locally built)",
                        'yes': "Showing only blocks from MEV relay",
                        'no': "Showing only locally built blocks"
                    }.get(config.get('mev_filter'), "")
                    if mev_info:
                        st.info(f"🎰 {mev_info}")
                
                # Show raw data preview
                if st.checkbox("Show raw data preview"):
                    st.dataframe(
                        data.head(100),
                        use_container_width=True
                    )
                
                # Debug MEV grouping
                if config.get('grouping_dimension') in ['node_type_mev', 'cl_node_type_mev']:
                    if st.checkbox("Debug: Show MEV grouping"):
                        if 'group_key' in data.columns:
                            st.write("Group key distribution:")
                            st.write(data['group_key'].value_counts())
                        if 'slot' in data.columns:
                            sample_slots = data['slot'].head(10).tolist()
                            st.write(f"Sample slots from data: {sample_slots}")
                            mev_slots_set = set(st.session_state.peerdas_v2_analysis_data.get('mev_slots', []))
                            st.write(f"MEV slots loaded: {len(mev_slots_set)}")
                            if mev_slots_set and sample_slots:
                                overlap = set(sample_slots).intersection(mev_slots_set)
                                st.write(f"Slots in both lists: {overlap}")
            
            # Debug section for node visibility
            with st.expander("🐛 Debug: Node Visibility", expanded=False):
                # Get node classifications from network YAML
                node_classifications = get_node_classifications(config['network'])
                
                if not node_classifications.empty:
                    # Filter proposer nodes based on criteria
                    proposer_nodes = node_classifications.copy()
                    if config.get('proposer_type') and config.get('proposer_type') != 'all':
                        proposer_nodes = proposer_nodes[proposer_nodes['node_type'] == config['proposer_type']]
                    if config.get('proposer_cl'):
                        proposer_nodes = proposer_nodes[proposer_nodes['cl_implementation'].isin(config['proposer_cl'])]
                    if config.get('proposer_el'):
                        proposer_nodes = proposer_nodes[proposer_nodes['el_implementation'].isin(config['proposer_el'])]
                    
                    # Filter attester nodes based on criteria
                    attester_nodes = node_classifications.copy()
                    if config.get('attester_type') and config.get('attester_type') != 'all':
                        attester_nodes = attester_nodes[attester_nodes['node_type'] == config['attester_type']]
                    if config.get('attester_cl'):
                        attester_nodes = attester_nodes[attester_nodes['cl_implementation'].isin(config['attester_cl'])]
                    if config.get('attester_el'):
                        attester_nodes = attester_nodes[attester_nodes['el_implementation'].isin(config['attester_el'])]
                    
                    # Get group keys that actually appeared in the data
                    actual_group_keys = set()
                    if 'group_key' in data.columns:
                        actual_group_keys = set(data['group_key'].dropna().unique())
                    
                    # Display debug information
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Proposer Nodes (from filters):**")
                        st.write(f"Total: {len(proposer_nodes)} nodes")
                        
                        # Sort by node type and name for better readability
                        proposer_sorted = proposer_nodes.sort_values(['node_type', 'client_name'])
                        
                        for _, node in proposer_sorted.iterrows():
                            node_name = node['client_name']
                            node_type = node['node_type']
                            
                            # Format node info
                            node_info = f"{node_name} ({node_type})"
                            
                            st.text(node_info)
                    
                    with col2:
                        st.write("**Attester Nodes (from filters):**")
                        st.write(f"Total: {len(attester_nodes)} nodes")
                        
                        # Sort by node type and name for better readability
                        attester_sorted = attester_nodes.sort_values(['node_type', 'client_name'])
                        
                        for _, node in attester_sorted.iterrows():
                            node_name = node['client_name']
                            node_type = node['node_type']
                            
                            # Format node info
                            node_info = f"{node_name} ({node_type})"
                            
                            st.text(node_info)
                    
                    # Show actual group keys found in data
                    st.write("---")
                    st.write(f"**Group Keys in Data (grouping by: {config.get('grouping_dimension')})**")
                    if actual_group_keys:
                        st.write(f"Found {len(actual_group_keys)} unique group keys:")
                        for key in sorted(actual_group_keys):
                            st.text(f"  • {key}")
                    else:
                        st.warning("No group keys found in data")
                    
                    # Show slot coverage
                    st.write("---")
                    st.write("**Slot Coverage:**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Eligible Slots", st.session_state.peerdas_v2_analysis_data.get('unique_slots', 0))
                    with col2:
                        st.metric("Analyzed Slots", st.session_state.peerdas_v2_analysis_data.get('total_slots_analyzed', 0))
                    with col3:
                        st.metric("Filtered Out", st.session_state.peerdas_v2_analysis_data.get('filtered_out_slots', 0))
                else:
                    st.warning("No node classifications found in network YAML file")
    


if __name__ == "__main__":
    main()
