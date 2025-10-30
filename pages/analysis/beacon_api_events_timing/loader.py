"""
Data loader for Beacon API Events Timing analysis.

Replicates grouping and filters style from peerdas_analysis_v2 but focuses on
timing diffs from beacon API event tables.
"""

from typing import Dict, Any, List
from datetime import datetime
import pandas as pd
import streamlit as st

from shared.database import get_database_connection
from shared.ethereum.validator_filters import get_node_classifications

from pages.analysis.beacon_api_events_timing.queries import (
    build_time_series_query,
    build_simple_samples_query,
    build_grouped_samples_query,
    BEACON_API_TABLES,
    LIBP2P_TABLES,
)


PROPOSER_GROUP_EXPRESSIONS = {
    'none': "'all'",
    'node_type': "coalesce(vm.node_type, 'unknown')",
    'cl_client': "coalesce(vm.cl_client, 'unknown')",
    'el_client': "coalesce(vm.el_client, 'unknown')",
    'architecture': "coalesce(vm.architecture, 'unknown')",
    'operator': "coalesce(vm.operator, 'unknown')",
    'region': "coalesce(vm.region, 'unknown')",
    'datacenter': "coalesce(vm.datacenter, 'unknown')",
    'cl_el_combined': "coalesce(concat(vm.cl_client, '-', vm.el_client), 'unknown')",
    'cl_node_type': "coalesce(concat(vm.cl_client, '-', vm.node_type), 'unknown')",
    'cl_architecture': "coalesce(concat(vm.cl_client, '-', vm.architecture), 'unknown')",
    'cl_operator': "coalesce(concat(vm.cl_client, '-', vm.operator), 'unknown')",
    'block_building': "'unknown'",
    'node_type_mev': "coalesce(concat(vm.node_type, '-', 'unknown'), 'unknown')",
    'cl_node_type_mev': "coalesce(concat(vm.cl_client, '-', vm.node_type, '-', 'unknown'), 'unknown')",
}

RECEIVER_GROUP_EXPRESSIONS = {
    'none': "'all'",
    'cl_client': "coalesce(vm.cl_client, 'unknown')",
    'client_instance': "coalesce(vm.client_instance, 'unknown')",
}


@st.cache_data(ttl=300, show_spinner=False, persist=False)
def get_unique_clients(cluster_name: str, network: str):
    # Returns a DataFrame of classifications
    return get_node_classifications(network, cluster_name)


@st.cache_data(ttl=120, show_spinner=True, persist=False)
def load_event_timing_grouped(
    cluster_name: str,
    network: str,
    start_date: datetime,
    end_date: datetime,
    data_source: str,
    event_type: str,
    proposer_grouping: str,
    receiver_grouping: str,
    performance_threshold_ms: int,
    sample_rate: int,
    max_records: int,
    proposer_filters: Dict[str, Any],
    receiver_filters: Dict[str, Any],
    enable_blob_bucketing: bool = False,
    mev_filter: str = 'both',
) -> Dict[str, Any]:
    conn = get_database_connection(cluster_name)
    if not conn:
        return {"time_series": pd.DataFrame(), "samples": pd.DataFrame()}

    # Time series overview (use limited records for time series to avoid memory issues)
    ts_max_records = 10000 if max_records != 0 else 0
    ts_sql = build_time_series_query(network, data_source, event_type, ts_max_records)
    ts_params = {"start_date": start_date, "end_date": end_date}
    ts_df = pd.read_sql(ts_sql, conn, params=ts_params)

    # Build grouping expressions
    prop_expr = PROPOSER_GROUP_EXPRESSIONS.get(proposer_grouping, "'all'")
    recv_expr = RECEIVER_GROUP_EXPRESSIONS.get(receiver_grouping, "'all'")

    # Build filter SQL snippets
    def _in_clause(values: List[str]) -> str:
        if not values:
            return ""
        safe = ",".join(f"'{v}'" for v in values)
        return f" IN ({safe})"

    prop_filter_sql_parts: List[str] = []
    if proposer_filters.get('proposer_type'):
        prop_filter_sql_parts.append(f"AND coalesce(vm.node_type, 'unknown') = '{proposer_filters['proposer_type']}'")
    if proposer_filters.get('proposer_cl'):
        prop_filter_sql_parts.append(f"AND coalesce(vm.cl_client, 'unknown'){_in_clause(proposer_filters['proposer_cl'])}")
    if proposer_filters.get('proposer_el'):
        prop_filter_sql_parts.append(f"AND coalesce(vm.el_client, 'unknown'){_in_clause(proposer_filters['proposer_el'])}")
    if proposer_filters.get('proposer_architecture'):
        prop_filter_sql_parts.append(f"AND coalesce(vm.architecture, 'unknown'){_in_clause(proposer_filters['proposer_architecture'])}")
    if proposer_filters.get('proposer_operator'):
        prop_filter_sql_parts.append(f"AND coalesce(vm.operator, 'unknown'){_in_clause(proposer_filters['proposer_operator'])}")
    if proposer_filters.get('proposer_region'):
        prop_filter_sql_parts.append(f"AND coalesce(vm.region, 'unknown'){_in_clause(proposer_filters['proposer_region'])}")
    if proposer_filters.get('proposer_datacenter'):
        prop_filter_sql_parts.append(f"AND coalesce(vm.datacenter, 'unknown'){_in_clause(proposer_filters['proposer_datacenter'])}")
    proposer_filter_sql = ("\n    " + "\n    ".join(prop_filter_sql_parts)) if prop_filter_sql_parts else ""

    recv_filter_sql_parts: List[str] = []
    # Event receiver filters using available beacon API metadata
    if receiver_filters.get('attester_cl'):
        recv_filter_sql_parts.append(f"AND coalesce(vm.cl_client, 'unknown'){_in_clause(receiver_filters['attester_cl'])}")
    receiver_filter_sql = ("\n    " + "\n    ".join(recv_filter_sql_parts)) if recv_filter_sql_parts else ""

    # Use grouped query when grouping is enabled, otherwise simple query
    if proposer_grouping != 'none' or receiver_grouping != 'none':
        samples_sql = build_grouped_samples_query(
            network=network,
            data_source=data_source,
            event=event_type,
            performance_threshold_ms=performance_threshold_ms,
            sample_rate=sample_rate,
            max_records=max_records,
            proposer_group_expr=prop_expr,
            receiver_group_expr=recv_expr,
            proposer_filter_sql=proposer_filter_sql,
            receiver_filter_sql=receiver_filter_sql,
            enable_blob_bucketing=enable_blob_bucketing,
            mev_filter=mev_filter,
        )
    else:
        samples_sql = build_simple_samples_query(
            network=network,
            data_source=data_source,
            event=event_type,
            performance_threshold_ms=performance_threshold_ms,
            sample_rate=sample_rate,
            max_records=max_records,
        )

    params = {"start_date": start_date, "end_date": end_date, "performance_threshold_ms": performance_threshold_ms}
    samples_df = pd.read_sql(samples_sql, conn, params=params)

    return {"time_series": ts_df, "samples": samples_df}


