"""
Interactive dashboard for Reorg Rates analysis.

Analyzes block reorganization rates by comparing proposed vs canonical blocks,
with filtering by proposer characteristics and MEV status.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import shared components
from shared.header import render_global_header, get_global_cluster, get_global_network
from shared.database import get_database_connection
from shared.ethereum.validator_filters import create_proposer_filters_ui

# Import local modules
from pages.analysis.reorg_rates.loader import (
    get_canonical_max_slot,
    load_eligible_slots,
    load_reorg_data,
    validate_canonical_data_availability
)
from pages.analysis.reorg_rates.plot_generators import (
    create_reorg_count_bar_chart,
    create_reorg_rate_chart,
    create_reorg_rate_boxplot,
    create_reorg_rate_violin,
    create_advanced_grouped_boxplot,
    create_reorg_rate_ecdf,
    create_reorg_rate_cdf
)


def initialize_session_state():
    """Initialize session state variables."""
    if 'reorg_rates_data_loaded' not in st.session_state:
        st.session_state.reorg_rates_data_loaded = False
    if 'reorg_rates_analysis_data' not in st.session_state:
        st.session_state.reorg_rates_analysis_data = {}
    if 'reorg_rates_last_config' not in st.session_state:
        st.session_state.reorg_rates_last_config = None


def render_sidebar_config(cluster: str, network: str) -> Dict[str, Any]:
    """
    Render sidebar configuration options.
    
    Returns:
        Configuration dictionary
    """
    st.sidebar.header("⚙️ Configuration")
    
    # Check if network changed to clear cache
    if 'reorg_rates_last_network' not in st.session_state:
        st.session_state.reorg_rates_last_network = network
    elif st.session_state.reorg_rates_last_network != network:
        st.session_state.reorg_rates_last_network = network
        st.session_state.reorg_rates_data_loaded = False
        st.session_state.reorg_rates_analysis_data = {}
        logger.info(f"Network changed to {network}, clearing cache")
    
    # Check if cluster changed
    if 'reorg_rates_last_cluster' not in st.session_state:
        st.session_state.reorg_rates_last_cluster = cluster
    elif st.session_state.reorg_rates_last_cluster != cluster:
        st.session_state.reorg_rates_last_cluster = cluster
        st.session_state.reorg_rates_data_loaded = False
        st.session_state.reorg_rates_analysis_data = {}
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
        
    # Grouping selection (moved to top)
    st.sidebar.subheader("🧩 Grouping")
    
    # Check if network has a spec for grouping
    from shared.network_spec import get_network_spec
    network_has_spec = get_network_spec(network) is not None
    
    if not network_has_spec:
        st.sidebar.warning("⚠️ Grouping not available for this network (no network spec)")
        grouping_dimension = None
    else:
        grouping_dimension = st.sidebar.selectbox(
            "Grouping Dimension",
            options=['none', 'node_type', 'cl_client', 'el_client', 'cl_el_combined', 'cl_node_type', 'block_building', 'node_type_mev', 'cl_node_type_mev'],
            format_func=lambda x: {
                'none': 'No Grouping',
                'node_type': 'Node Type',
                'cl_client': 'CL Client',
                'el_client': 'EL Client',
                'cl_el_combined': 'CL+EL Combination',
                'cl_node_type': 'CL+Node Type',
                'block_building': 'Block Building Method',
                'node_type_mev': 'Node Type + MEV',
                'cl_node_type_mev': 'CL + Node Type + MEV'
            }[x],
            index=1,  # Default to node_type
            help="Group reorg rates by proposer characteristics"
        )
        if grouping_dimension == 'none':
            grouping_dimension = None
    
    # Time range selection
    st.sidebar.subheader("🕒 Time Range")
    
    # Preset time ranges
    time_preset = st.sidebar.selectbox(
        "Quick Select",
        options=["last_1h", "last_6h", "last_12h", "last_1d", "last_3d", "last_7d", "last_14d", "last_31d", "last_90d", "custom"],
        format_func=lambda x: {
            "last_1h": "Last 1 hour",
            "last_6h": "Last 6 hours", 
            "last_12h": "Last 12 hours",
            "last_1d": "Last 24 hours",
            "last_3d": "Last 3 days",
            "last_7d": "Last 7 days",
            "last_14d": "Last 14 days",
            "last_31d": "Last 31 days",
            "last_90d": "Last 90 days",
            "custom": "Custom range"
        }[x],
        index=3,  # Default to "Last 24 hours"
        help="Select a predefined time range or choose custom"
    )
    
    # Calculate dates based on preset
    if time_preset != "custom":
        end_datetime = datetime.now()
        if time_preset == "last_1h":
            start_datetime = end_datetime - timedelta(hours=1)
        elif time_preset == "last_6h":
            start_datetime = end_datetime - timedelta(hours=6)
        elif time_preset == "last_12h":
            start_datetime = end_datetime - timedelta(hours=12)
        elif time_preset == "last_1d":
            start_datetime = end_datetime - timedelta(days=1)
        elif time_preset == "last_3d":
            start_datetime = end_datetime - timedelta(days=3)
        elif time_preset == "last_7d":
            start_datetime = end_datetime - timedelta(days=7)
        elif time_preset == "last_14d":
            start_datetime = end_datetime - timedelta(days=14)
        elif time_preset == "last_31d":
            start_datetime = end_datetime - timedelta(days=31)
        elif time_preset == "last_90d":
            start_datetime = end_datetime - timedelta(days=90)
        
        # Show the selected range (read-only)
        st.sidebar.text(f"From: {start_datetime.strftime('%Y-%m-%d %H:%M')}")
        st.sidebar.text(f"To:   {end_datetime.strftime('%Y-%m-%d %H:%M')}")
    else:
        # Custom date selection
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now().date() - timedelta(days=1),
                help="Start date for analysis"
            )
        
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now().date(),
                help="End date for analysis"
            )
        
        # Convert to datetime
        start_datetime = datetime.combine(start_date, datetime.min.time())
        end_datetime = datetime.combine(end_date, datetime.max.time())
    
    # MEV filter selection
    st.sidebar.subheader("💰 MEV Filter")
    mev_filter = st.sidebar.selectbox(
        "MEV Filter",
        options=['both', 'yes', 'no'],
        format_func=lambda x: {
            'both': 'All Blocks (MEV + Local)',
            'yes': 'MEV Relay Only',
            'no': 'Local Build Only'
        }[x],
        index=0,
        help="Filter by block building method"
    )
    
    # Proposer filters (no attester filters for reorg analysis)
    with st.sidebar:
        proposer_filters = create_proposer_filters_ui(network, key_prefix="reorg_proposer")
    
    # Unknown validator filter
    st.sidebar.subheader("🔍 Data Filters")
    ignore_unknown = st.sidebar.checkbox(
        "Ignore unknown validators",
        value=False,
        help="Exclude validators not defined in the network specification from analysis"
    )
    
    # Chart configuration
    st.sidebar.subheader("📊 Chart Settings")
    
    chart_type = st.sidebar.selectbox(
        "Chart Type",
        options=['bar', 'scatter', 'time_series', 'heatmap'],
        format_func=lambda x: {
            'bar': 'Bar Chart (Reorg Counts)',
            'scatter': 'Scatter Plot',
            'time_series': 'Time Series',
            'heatmap': 'Heatmap by Hour/Day'
        }[x]
    )
    
    num_buckets = st.sidebar.slider("Number of Buckets", 3, 20, 6, help="Number of buckets for grouping data")
    
    show_trend_line = False
    scatter_aggregation = 'p95'
    if chart_type == 'scatter':
        show_trend_line = st.sidebar.checkbox(
            "Show trend line",
            value=True,
            help="Display trend line on scatter plot"
        )
        
        scatter_aggregation = st.sidebar.selectbox(
            "Aggregation Method",
            options=['mean', 'median', 'p25', 'p50', 'p75', 'p90', 'p95', 'p99', 'min', 'max'],
            index=6,  # Default to p95
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
            help="Aggregation method for scatter plot"
        )
    
    return {
        'cluster': cluster,
        'start_datetime': start_datetime,
        'end_datetime': end_datetime,
        'grouping_dimension': grouping_dimension,
        'mev_filter': mev_filter,
        'proposer_filters': proposer_filters,
        'ignore_unknown': ignore_unknown,
        'num_buckets': num_buckets,
        'chart_type': chart_type,
        'show_trend_line': show_trend_line,
        'scatter_aggregation': scatter_aggregation
    }


def render_info_section(config: Dict[str, Any]):
    """Render information section about the analysis."""
    st.info("""
    📊 **Reorg Rates Analysis**
    
    This analysis identifies block reorganizations by comparing:
    - **Proposed blocks** (`beacon_api_eth_v2_beacon_block`): All blocks that were proposed
    - **Canonical blocks** (`canonical_beacon_block`): Blocks that became part of the finalized chain
    
    **⚠️ Important:** Only finalized blocks are analyzed to avoid false positives from recent unfinalized blocks.
    The canonical table is typically 15+ minutes behind the head of the chain.
    """)


def load_and_process_data(config: Dict[str, Any], network: str) -> Optional[pd.DataFrame]:
    """Load and process reorg rate data."""
    logger.info("Starting data loading process")
    
    # Check canonical data availability first
    availability = validate_canonical_data_availability(
        network=network,
        start_date=config['start_datetime'],
        end_date=config['end_datetime'],
        cluster_name=config['cluster']
    )
    
    if not availability['available']:
        st.error(f"""
        ❌ **Canonical block data not available**
        
        {availability.get('error', 'Unknown error')}
        
        Reorg analysis requires canonical block data to determine which blocks 
        were finalized vs reorged. Please try a different time range or network.
        """)
        return None
    
    # Show warnings about canonical table lag
    if 'warnings' in availability and availability['warnings']:
        for warning in availability['warnings']:
            st.warning(f"⚠️ {warning}")
    
    # Show data availability info
    if 'canonical_blocks' in availability and 'proposed_blocks' in availability:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Canonical Blocks", f"{availability['canonical_blocks']:,}")
        with col2:
            st.metric("Proposed Blocks", f"{availability['proposed_blocks']:,}")
        with col3:
            if availability['canonical_blocks'] > 0 and availability['proposed_blocks'] > 0:
                potential_reorgs = availability['proposed_blocks'] - availability['canonical_blocks']
                st.metric("Potential Reorgs", f"{potential_reorgs:,}")
    
    # Get canonical max slot to limit analysis
    max_canonical_slot = get_canonical_max_slot(
        network=network,
        start_date=config['start_datetime'],
        end_date=config['end_datetime'],
        cluster_name=config['cluster']
    )
    
    if max_canonical_slot is None:
        st.error("❌ Could not determine canonical max slot")
        return None
    
    # Load eligible slots
    eligible_slots, slot_to_block, slot_to_proposer, mev_slots = load_eligible_slots(
        network=network,
        start_date=config['start_datetime'],
        end_date=config['end_datetime'],
        max_canonical_slot=max_canonical_slot,
        proposer_type=config['proposer_filters']['proposer_type'],
        cl_filter=config['proposer_filters']['proposer_cl'],
        el_filter=config['proposer_filters']['proposer_el'],
        mev_filter=config['mev_filter'],
        cluster_name=config['cluster']
    )
    
    if not eligible_slots:
        st.error("❌ No eligible slots found for the specified criteria")
        return None
    
    st.success(f"✅ Found {len(eligible_slots):,} eligible finalized slots")
    
    # Load reorg data
    with st.spinner("Loading reorg rate data..."):
        df = load_reorg_data(
            network=network,
            start_date=config['start_datetime'],
            end_date=config['end_datetime'],
            eligible_slots=eligible_slots,
            slot_to_block=slot_to_block,
            slot_to_proposer=slot_to_proposer,
            mev_slots=mev_slots,
            max_canonical_slot=max_canonical_slot,
            proposer_type=config['proposer_filters']['proposer_type'],
            cl_filter=config['proposer_filters']['proposer_cl'],
            el_filter=config['proposer_filters']['proposer_el'],
            grouping_dimension=config['grouping_dimension'],
            cluster_name=config['cluster']
        )
    
    if df.empty:
        st.error("❌ No reorg data could be loaded")
        return None
    
    logger.info(f"Loaded {len(df)} rows of reorg data")
    
    # Filter out unknown validators if requested (only when grouping is enabled)
    if config.get('ignore_unknown', False) and config.get('grouping_dimension') and 'group_key' in df.columns:
        before_count = len(df)
        df = df[df['group_key'] != 'unknown']
        after_count = len(df)
        if before_count != after_count:
            filtered_count = before_count - after_count
            logger.info(f"Filtered out {filtered_count} rows with unknown validators")
            st.info(f"Filtered out {filtered_count} blocks from unknown validators")
            
            # Check if we still have data after filtering
            if df.empty:
                st.warning("⚠️ No data remaining after filtering out unknown validators")
                return None
    
    return df


def render_analysis_results(df: pd.DataFrame, config: Dict[str, Any], network: str):
    """Render analysis results with charts and metrics."""
    if df.empty:
        st.warning("No data to display")
        return
    
    # Calculate summary metrics
    total_blocks = len(df)
    reorged_blocks = len(df[df['is_reorged'] == 1])
    reorg_rate = (reorged_blocks / total_blocks * 100) if total_blocks > 0 else 0
    
    # Display summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Blocks", f"{total_blocks:,}")
    with col2:
        st.metric("Reorged Blocks", f"{reorged_blocks:,}")
    with col3:
        st.metric("Reorg Rate", f"{reorg_rate:.2f}%")
    
    # Show sample of reorged blocks if any
    if reorged_blocks > 0:
        with st.expander(f"🔍 Sample Reorged Blocks ({min(10, reorged_blocks)} of {reorged_blocks})"):
            reorged_sample = df[df['is_reorged'] == 1].head(10)[['slot', 'proposer_index', 'proposed_block_root']]
            st.dataframe(reorged_sample, use_container_width=True)
    
    # Create time range string for chart titles
    time_range = f"{config['start_datetime'].strftime('%Y-%m-%d')} to {config['end_datetime'].strftime('%Y-%m-%d')}"
    
    # Metadata for charts
    metadata = {
        'network': network,
        'time_range': time_range,
        'total_blocks': total_blocks,
        'reorged_blocks': reorged_blocks,
        'mev_filter': config['mev_filter']
    }
    
    # Create chart based on selected type
    if config['chart_type'] == 'bar':
        try:
            fig = create_reorg_count_bar_chart(
                data=df,
                grouping_dimension=config['grouping_dimension'],
                title_suffix="",
                network=network,
                time_range=time_range,
                metadata=metadata,
                proposer_filters=config['proposer_filters']
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating bar chart: {e}")
            
    elif config['chart_type'] == 'time_series':
        try:
            # Create time series plot showing reorg rate over time
            fig = create_reorg_rate_chart(
                data=df,
                num_buckets=config['num_buckets'],
                title_suffix=" - Time Series",
                network=network,
                time_range=time_range,
                metadata=metadata,
                show_trend_line=config['show_trend_line'],
                aggregation_method=config.get('scatter_aggregation', 'mean'),
                grouping_dimension=config['grouping_dimension'],
                proposer_filters=config['proposer_filters']
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating time series: {e}")
            
    elif config['chart_type'] == 'heatmap':
        try:
            st.info("Heatmap visualization coming soon - will show reorg patterns by hour of day and day of week")
        except Exception as e:
            st.error(f"Error creating heatmap: {e}")
            
    else:  # Default to scatter plot
        try:
            fig = create_reorg_rate_chart(
                data=df,
                num_buckets=config['num_buckets'],
                title_suffix="",
                network=network,
                time_range=time_range,
                metadata=metadata,
                show_trend_line=config['show_trend_line'],
                aggregation_method=config.get('scatter_aggregation', 'p95'),
                grouping_dimension=config['grouping_dimension'],
                proposer_filters=config['proposer_filters']
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating scatter plot: {e}")


def main():
    """Main dashboard function."""
    # Initialize session state
    initialize_session_state()
    
    # Render global header
    render_global_header()
    
    # Get global settings
    cluster = get_global_cluster()
    network = get_global_network()
    
    # Render sidebar configuration
    config = render_sidebar_config(cluster, network)
    
    # Check if configuration changed
    config_changed = st.session_state.reorg_rates_last_config != config
    if config_changed:
        st.session_state.reorg_rates_last_config = config.copy()
        st.session_state.reorg_rates_data_loaded = False
        st.session_state.reorg_rates_analysis_data = {}
        logger.info("Configuration changed, clearing cache")
    
    # Load and Clear Cache buttons in sidebar
    st.sidebar.markdown("---")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        load_button = st.sidebar.button("🔄 Load Data", type="primary", use_container_width=True)
    with col2:
        if st.sidebar.button("🗑️ Clear", use_container_width=True):
            st.cache_data.clear()
            st.session_state.reorg_rates_analysis_data = {}
            st.session_state.reorg_rates_data_loaded = False
            st.rerun()
    
    if load_button:
        # Load and process data
        df = load_and_process_data(config, network)
        
        if df is not None:
            st.session_state.reorg_rates_analysis_data = df
            st.session_state.reorg_rates_data_loaded = True
            logger.info("Data loaded and cached successfully")
        else:
            st.session_state.reorg_rates_data_loaded = False
            return
    
    # Render results if data is loaded
    if st.session_state.reorg_rates_data_loaded and not st.session_state.reorg_rates_analysis_data.empty:
        render_analysis_results(st.session_state.reorg_rates_analysis_data, config, network)


if __name__ == "__main__":
    main()