"""
Interactive dashboard for Beacon API Events Timing analysis.

Mirrors the configuration and grouping patterns from peerdas_analysis_v2, but
operates on beacon API event timing metrics, allowing selection of a specific
event type and aggregating timing by proposer/attester groupings.
"""

import streamlit as st
from datetime import datetime, timedelta, timezone
from typing import Dict, Any

from shared.header import render_global_header, get_global_cluster, get_global_network
from shared.ethereum.validator_filters import (
    create_proposer_filters_ui,
    create_attester_filters_ui,
)

from pages.analysis.beacon_api_events_timing.loader import (
    load_event_timing_grouped,
    get_unique_clients,
)

from pages.analysis.beacon_api_events_timing.plot_generators import (
    render_time_series_summary,
    render_group_boxplots,
    render_data_summary,
)

from pages.analysis.beacon_api_events_timing.chart_functions import (
    render_histogram_analysis,
    render_statistical_summary,
)


def render_sidebar_config(cluster: str, network: str) -> Dict[str, Any]:
    st.sidebar.header("⚙️ Configuration")

    # Time range
    st.sidebar.subheader("📅 Time Range")
    selected_period = st.sidebar.selectbox(
        "Quick period",
        ["Last 1 Hour", "Last 2 Hours", "Last 6 Hours", "Last 12 Hours", "Last 24 Hours", "Custom"],
        index=1,
    )

    if selected_period != "Custom":
        hours = {
            "Last 1 Hour": 1,
            "Last 2 Hours": 2,
            "Last 6 Hours": 6,
            "Last 12 Hours": 12,
            "Last 24 Hours": 24,
        }[selected_period]
        end_datetime = datetime.now(timezone.utc).replace(tzinfo=None)
        start_datetime = end_datetime - timedelta(hours=hours)
    else:
        default_end = datetime.now(timezone.utc).replace(tzinfo=None)
        default_start = default_end - timedelta(hours=2)

        st.sidebar.subheader("Start Time")
        start_col1, start_col2 = st.sidebar.columns(2)
        start_date = start_col1.date_input(
            "Start Date",
            value=st.session_state.get('beapi_start_date_custom', default_start.date()),
            max_value=datetime.now().date(),
            key="beapi_start_date_custom"
        )
        start_time = start_col2.time_input(
            "Start Time (UTC)",
            value=st.session_state.get('beapi_start_time_custom', default_start.time()),
            key="beapi_start_time_custom",
            step=300
        )

        st.sidebar.subheader("End Time")
        end_col1, end_col2 = st.sidebar.columns(2)
        end_date = end_col1.date_input(
            "End Date",
            value=st.session_state.get('beapi_end_date_custom', default_end.date()),
            max_value=datetime.now().date(),
            key="beapi_end_date_custom"
        )
        end_time = end_col2.time_input(
            "End Time (UTC)",
            value=st.session_state.get('beapi_end_time_custom', default_end.time()),
            key="beapi_end_time_custom",
            step=300
        )

        start_datetime = datetime.combine(start_date, start_time)
        end_datetime = datetime.combine(end_date, end_time)

    # Event type selection
    st.sidebar.subheader("📊 Data Source")
    data_source = st.sidebar.selectbox(
        "Data Source",
        options=[
            "beacon_api",
            "libp2p_gossipsub",
        ],
        index=0,
        help="Select data source: Beacon API events or LibP2P Gossipsub messages",
    )

    if data_source == "beacon_api":
        event_type = st.sidebar.selectbox(
            "Beacon API Event",
            options=[
                "block",
                "head",
                "blob_sidecar",
                "attestation",
                "sync_committee",
            ],
            index=0,
            help="Select the event type table to aggregate timing from",
        )
    else:
        event_type = st.sidebar.selectbox(
            "LibP2P Event",
            options=[
                "beacon_block",
                "beacon_attestation",
                "data_column_sidecar",
                "blob_sidecar",
            ],
            index=0,
            help="Select the libp2p gossipsub message type to analyze",
        )

    # Grouping selections (matching peerdas_analysis_v2 format)
    st.sidebar.subheader("🧩 Grouping")

    # Proposer grouping (primary for beacon API events)
    proposer_grouping = st.sidebar.selectbox(
        "Proposer Grouping",
        options=['none', 'node_type', 'cl_client', 'el_client', 'architecture', 'operator', 'region', 'datacenter', 'cl_el_combined', 'cl_node_type', 'cl_architecture', 'cl_operator'],
        index=1,  # Default to 'node_type'
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
            'cl_operator': 'CL+Operator'
        }[x],
        help="Group events by proposer characteristics"
    )

    # Event Receiver grouping (the client that received and reported the event)
    receiver_grouping = st.sidebar.selectbox(
        "Event Receiver Grouping",
        options=['none', 'cl_client', 'client_instance'],
        index=1,  # Default to 'cl_client'
        format_func=lambda x: {
            'none': 'None (All Receivers)',
            'cl_client': 'CL Client Type',
            'client_instance': 'Client Instance',
        }[x],
        help="Group events by receiver characteristics. CL client type uses meta_client_implementation, client instance shows individual nodes."
    )

    # Filtering section (matching peerdas_analysis_v2 structure)
    st.sidebar.markdown("---")

    # Create filter UI components using shared utility
    with st.sidebar:
        proposer_filters = create_proposer_filters_ui(
            network=network,
            cluster_name=cluster,
            key_prefix="beapi_proposer",
            initial_values=None,
        )

        # Event receiver filters (using available beacon API metadata)
        st.subheader("🎯 Event Receiver Filters")

        receiver_cl_options = ['all', 'lighthouse', 'prysm', 'teku', 'nimbus', 'lodestar', 'grandine']
        receiver_cl = st.multiselect(
            "Receiver CL Client",
            options=receiver_cl_options,
            default=['all'],
            help="Filter by consensus client type that received the event"
        )

        receiver_filters = {
            'attester_cl': receiver_cl if receiver_cl != ['all'] else []
        }

    # Sampling controls to prevent message size errors
    st.sidebar.subheader("🎛️ Sampling & Limits")

    # Calculate approximate data volume based on time range
    time_diff = end_datetime - start_datetime
    hours_selected = time_diff.total_seconds() / 3600

    # Unlimited mode checkbox - define early so it can be used below
    unlimited_mode = st.sidebar.checkbox(
        "Unlimited Mode",
        value=False,
        help="Remove all record limits - query ALL available data (may take 5-15 minutes for large datasets)"
    )

    # Suggest sampling based on time range
    if hours_selected <= 1:
        suggested_sample = 100  # Full data for 1 hour or less
        suggested_max = 100000
    elif hours_selected <= 6:
        suggested_sample = 50   # 50% for 1-6 hours
        suggested_max = 50000
    elif hours_selected <= 12:
        suggested_sample = 25   # 25% for 6-12 hours
        suggested_max = 25000
    else:
        suggested_sample = 10   # 10% for 12+ hours
        suggested_max = 10000

    if not unlimited_mode:  # Only show sampling controls when not in unlimited mode
        use_sampling = st.sidebar.checkbox(
            "Enable Data Sampling",
            value=(hours_selected > 6),  # Auto-enable for >6 hour ranges
            help="Uncheck to query all data (may be slow for large time ranges)"
        )

        if use_sampling:
            sample_rate = st.sidebar.slider(
                "Sample Rate (%)",
                min_value=1,
                max_value=100,
                value=suggested_sample,
                help=f"Suggested: {suggested_sample}% for {hours_selected:.1f} hour range."
            )
        else:
            sample_rate = 100  # Full dataset

    if unlimited_mode:
        max_records = 0  # 0 means no limit in our queries
        sample_rate = 100  # Override any sampling settings
        st.sidebar.error("🚨 UNLIMITED MODE: Querying ALL data without limits")
        st.sidebar.warning("⚠️ This may take 5-15 minutes and consume significant memory for large time ranges")
        st.sidebar.info("ℹ️ Sampling automatically disabled in unlimited mode")

        # Performance optimization for unlimited mode
        st.sidebar.subheader("🛠️ Performance Options")
        use_aggregation = st.sidebar.checkbox(
            "Use Server-Side Aggregation",
            value=True,
            help="Pre-aggregate data on server to reduce memory usage"
        )

        if use_aggregation:
            st.sidebar.info("ℹ️ Data will be pre-aggregated by time windows to optimize performance")
    else:
        max_records = st.sidebar.number_input(
            "Max Records",
            min_value=1000,
            max_value=5000000,  # Increased limit
            value=suggested_max if use_sampling else 500000,
            help=f"Suggested: {suggested_max:,} for {hours_selected:.1f} hour range. Max: 5M records."
        )

    # Performance threshold - allow higher values for unlimited mode
    max_threshold = 600000 if unlimited_mode else 300000
    default_threshold = 300000 if unlimited_mode else 100000

    performance_threshold_ms = st.sidebar.number_input(
        "Outlier cap (ms)",
        min_value=0,
        max_value=max_threshold,
        value=default_threshold,
        help=f"Ignore diffs above this threshold. Max: {max_threshold:,}ms {'(increased for unlimited mode)' if unlimited_mode else ''}"
    )

    # Outlier Controls
    st.sidebar.subheader("📈 Outlier Handling")

    outlier_method = st.sidebar.selectbox(
        "Outlier Filtering",
        options=['none', 'iqr', 'percentile', 'zscore'],
        index=1,  # Default to IQR
        format_func=lambda x: {
            'none': 'No Filtering',
            'iqr': 'IQR Method (1.5×IQR)',
            'percentile': 'Percentile Capping',
            'zscore': 'Z-Score Method'
        }[x],
        help="Method to handle outliers in visualizations"
    )

    if outlier_method == 'percentile':
        outlier_percentile = st.sidebar.slider(
            "Outlier Percentile",
            min_value=90,
            max_value=99,
            value=95,
            help="Cap outliers above this percentile"
        )
    else:
        outlier_percentile = 95

    if outlier_method == 'zscore':
        zscore_threshold = st.sidebar.number_input(
            "Z-Score Threshold",
            min_value=1.0,
            max_value=5.0,
            value=3.0,
            step=0.5,
            help="Remove values beyond this many standard deviations"
        )
    else:
        zscore_threshold = 3.0

    show_outliers_toggle = st.sidebar.checkbox(
        "Show Outliers in Boxplots",
        value=False,
        help="Toggle outlier points display in boxplot visualizations"
    )

    # Blob Count Bucketing
    st.sidebar.subheader("🗂️ Blob Count Bucketing")

    enable_blob_bucketing = st.sidebar.checkbox(
        "Enable Blob Bucketing",
        value=False,
        help="Group timing analysis by number of blobs in each slot"
    )

    if enable_blob_bucketing:
        filter_zero_blobs = st.sidebar.checkbox(
            "Filter out 0 blob slots",
            value=True,
            help="Exclude slots with 0 blobs from the analysis"
        )

        num_buckets = st.sidebar.slider(
            "Number of Buckets",
            min_value=1,
            max_value=12,
            value=6,
            help="Number of buckets to divide blob counts into"
        )
    else:
        filter_zero_blobs = False
        num_buckets = 1

    # MEV/Block Building Filter
    st.sidebar.subheader("🎰 Block Building Filter")

    mev_filter = st.sidebar.selectbox(
        "Block Source",
        options=['both', 'yes', 'no'],
        index=0,  # Default to 'both'
        format_func=lambda x: {
            'both': 'All Blocks',
            'yes': 'MEV Relay Blocks Only',
            'no': 'Local Build Blocks Only'
        }[x],
        help="Filter blocks by their building method: MEV relay-built vs locally-built blocks"
    )

    # Chart Options (matching peerdas_analysis_v2)
    st.sidebar.subheader("📊 Chart Options")

    chart_type = st.sidebar.selectbox(
        "Chart Type",
        options=['timeseries', 'boxplot', 'histogram', 'summary'],
        index=0,
        format_func=lambda x: {
            'timeseries': 'Time Series + Boxplot',
            'boxplot': 'Boxplot Distribution',
            'histogram': 'Histogram Distribution',
            'summary': 'Statistical Summary'
        }[x],
        help="Choose the visualization type for event timing analysis"
    )

    # Action buttons with safety warnings for unlimited mode
    st.sidebar.markdown("---")

    if unlimited_mode:
        st.sidebar.subheader("⚠️ Safety Check")
        safety_confirmed = st.sidebar.checkbox(
            f"I understand this will query {hours_selected:.1f} hours of data without limits",
            value=False,
            help="Confirm you want to proceed with unlimited data query"
        )
        if not safety_confirmed:
            st.sidebar.error("❌ Please confirm above to enable unlimited query")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        button_disabled = unlimited_mode and not safety_confirmed
        button_text = "🚀 Query All Data" if unlimited_mode else "🚀 Analyze Now"
        load_data = st.sidebar.button(
            button_text,
            type="primary",
            use_container_width=True,
            disabled=button_disabled
        )
    with col2:
        if st.sidebar.button("🗑️ Clear", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    return {
        'start_date': start_datetime,
        'end_date': end_datetime,
        'data_source': data_source,
        'event_type': event_type,
        'proposer_grouping': proposer_grouping,
        'receiver_grouping': receiver_grouping,
        'performance_threshold_ms': performance_threshold_ms,
        'sample_rate': sample_rate,
        'max_records': max_records,
        'chart_type': chart_type,
        'period': selected_period if selected_period != "Custom" else None,
        'proposer_filters': proposer_filters,
        'receiver_filters': receiver_filters,
        'outlier_method': outlier_method,
        'outlier_percentile': outlier_percentile,
        'zscore_threshold': zscore_threshold,
        'show_outliers_toggle': show_outliers_toggle,
        'enable_blob_bucketing': enable_blob_bucketing,
        'filter_zero_blobs': filter_zero_blobs,
        'num_buckets': num_buckets,
        'mev_filter': mev_filter,
        'unlimited_mode': unlimited_mode,
        'load_data': load_data,
    }


def main():
    render_global_header()
    cluster = get_global_cluster()
    network = get_global_network()

    st.title("Beacon API Events Timing")

    config = render_sidebar_config(cluster=cluster, network=network)

    if not config.get('load_data'):
        if config.get('unlimited_mode'):
            st.info("Configure unlimited mode settings in the sidebar and confirm safety check.")
        else:
            st.info("Adjust settings in the sidebar and click 'Analyze Now'.")
        return

    # Enhanced loading message for unlimited mode
    if config.get('unlimited_mode'):
        time_range = config['end_date'] - config['start_date']
        hours = time_range.total_seconds() / 3600
        spinner_text = f"🚀 UNLIMITED MODE: Querying ALL data for {hours:.1f} hours... This may take 5-15 minutes."
        st.warning(f"⚠️ Processing unlimited dataset covering {hours:.1f} hours. Please be patient...")
    else:
        spinner_text = "Loading data..."

    with st.spinner(spinner_text):
        data = load_event_timing_grouped(
            cluster_name=cluster,
            network=network,
            start_date=config['start_date'],
            end_date=config['end_date'],
            data_source=config['data_source'],
            event_type=config['event_type'],
            proposer_grouping=config['proposer_grouping'],
            receiver_grouping=config['receiver_grouping'],
            performance_threshold_ms=config['performance_threshold_ms'],
            sample_rate=config['sample_rate'],
            max_records=config['max_records'],
            proposer_filters=config['proposer_filters'],
            receiver_filters=config['receiver_filters'],
            enable_blob_bucketing=config['enable_blob_bucketing'],
            mev_filter=config.get('mev_filter', 'both'),
        )

    # Data Summary Section
    render_data_summary(data, config)

    # Render charts based on selected type
    chart_type = config.get('chart_type', 'timeseries')

    if chart_type == 'timeseries':
        # Time series + grouped analysis (default)
        render_time_series_summary(data, config)
        render_group_boxplots(data, config)
    elif chart_type == 'boxplot':
        # Focus on boxplot distribution only
        render_group_boxplots(data, config)
    elif chart_type == 'histogram':
        # Focus on histogram distribution
        render_histogram_analysis(data, config)
    elif chart_type == 'summary':
        # Statistical summary table
        render_statistical_summary(data, config)


if __name__ == "__main__":
    main()


