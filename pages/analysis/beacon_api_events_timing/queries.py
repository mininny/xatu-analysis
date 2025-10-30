"""
Query builders for Beacon API Events Timing analysis.

Provides per-event table mapping and grouped aggregations mirroring peerdas v2
grouping logic, but returning timing statistics for selected events.
"""

from typing import Literal


EventType = Literal[
    "block", "head", "blob_sidecar", "attestation", "sync_committee"
]


BEACON_API_TABLES = {
    "block": "beacon_api_eth_v1_events_block",
    "head": "beacon_api_eth_v1_events_head",
    "blob_sidecar": "beacon_api_eth_v1_events_blob_sidecar",
    "attestation": "beacon_api_eth_v1_events_attestation",
    "sync_committee": "beacon_api_eth_v1_events_sync_committee"
}

LIBP2P_TABLES = {
    "beacon_block": "libp2p_gossipsub_beacon_block",
    "beacon_attestation": "libp2p_gossipsub_beacon_attestation",
    "data_column_sidecar": "libp2p_gossipsub_data_column_sidecar",
    "blob_sidecar": "libp2p_gossipsub_blob_sidecar"
}


BEACON_API_DIFF_COLUMN = {
    # All diffs are measured as milliseconds vs slot start from Xatu ingestion
    "block": "propagation_slot_start_diff",
    "head": "propagation_slot_start_diff",
    "blob_sidecar": "propagation_slot_start_diff",
    "attestation": "propagation_slot_start_diff",
    "sync_committee": "propagation_slot_start_diff",
}

LIBP2P_DIFF_COLUMN = {
    # LibP2P gossipsub propagation timing columns
    "beacon_block": "propagation_slot_start_diff",
    "beacon_attestation": "propagation_slot_start_diff",
    "data_column_sidecar": "propagation_slot_start_diff",
    "blob_sidecar": "propagation_slot_start_diff",
}


def _build_mev_filter_clause(mev_filter: str, mev_status_expr: str) -> str:
    """Build MEV filter SQL clause based on filter type.

    Args:
        mev_filter: 'both', 'yes' (MEV only), or 'no' (local build only)
        mev_status_expr: SQL expression that evaluates to mev status (e.g., "if(mev.slot > 0, 'mev', 'non-mev')")
    """
    if mev_filter == 'yes':
        return f"\n  AND ({mev_status_expr}) = 'mev'"
    elif mev_filter == 'no':
        return f"\n  AND ({mev_status_expr}) = 'non-mev'"
    return ""  # 'both' or None means no filter


def build_time_series_query(network: str, data_source: str, event: str, max_records: int = 10000) -> str:
    if data_source == "beacon_api":
        table = BEACON_API_TABLES[event]
        diff_col = BEACON_API_DIFF_COLUMN[event]
    else:  # libp2p_gossipsub
        table = LIBP2P_TABLES[event]
        diff_col = LIBP2P_DIFF_COLUMN[event]

    # Use fixed 5-minute buckets; caller filters with params
    return f"""
SELECT
  toStartOfFiveMinute(slot_start_date_time) as time,
  max({diff_col}) as max,
  quantile(0.95)({diff_col}) as p95,
  avg({diff_col}) as average,
  quantile(0.50)({diff_col}) as p50,
  quantile(0.05)({diff_col}) as p05,
  min({diff_col}) as min
FROM `{network}`.{table}
WHERE
  slot_start_date_time BETWEEN %(start_date)s AND %(end_date)s
GROUP BY time
ORDER BY time ASC{f'\nLIMIT {max_records}' if max_records > 0 else ''}
"""


def build_simple_samples_query(
    network: str,
    data_source: str,
    event: str,
    performance_threshold_ms: int,
    sample_rate: int,
    max_records: int,
) -> str:
    if data_source == "beacon_api":
        table = BEACON_API_TABLES[event]
        diff_col = BEACON_API_DIFF_COLUMN[event]
    else:  # libp2p_gossipsub
        table = LIBP2P_TABLES[event]
        diff_col = LIBP2P_DIFF_COLUMN[event]

    # Simple query with optional sampling
    sampling_clause = f"AND rand() %% 100 < {sample_rate}" if sample_rate < 100 else ""

    return f"""
SELECT
  slot,
  {diff_col} AS diff_ms,
  slot_start_date_time,
  'general' AS category
FROM `{network}`.{table}
WHERE slot_start_date_time BETWEEN %(start_date)s AND %(end_date)s
  AND {diff_col} < %(performance_threshold_ms)s
  AND {diff_col} IS NOT NULL
  {sampling_clause}
ORDER BY slot_start_date_time DESC{f'\nLIMIT {max_records}' if max_records > 0 else ''}
"""


def build_grouped_samples_query(
    network: str,
    data_source: str,
    event: str,
    performance_threshold_ms: int,
    sample_rate: int,
    max_records: int,
    proposer_group_expr: str,
    receiver_group_expr: str,
    proposer_filter_sql: str,
    receiver_filter_sql: str,
    enable_blob_bucketing: bool = False,
    mev_filter: str = 'both',
) -> str:
    if data_source == "beacon_api":
        table = BEACON_API_TABLES[event]
        diff_col = BEACON_API_DIFF_COLUMN[event]
    else:  # libp2p_gossipsub
        table = LIBP2P_TABLES[event]
        diff_col = LIBP2P_DIFF_COLUMN[event]

    # Enhanced query with both proposer and receiver grouping
    # Proposer = validator who proposed the block for this slot
    # Receiver = node that received and reported this beacon API event (identified by node_id in the event)

    blob_count_cte = ""
    if enable_blob_bucketing:
        blob_count_cte = f"""
blob_counts AS (
  SELECT b1.slot AS slot, toUInt64(length(anyLast(b1.kzg_commitments))) AS blob_count
  FROM `{network}`.beacon_api_eth_v1_events_data_column_sidecar AS b1
  WHERE b1.slot_start_date_time BETWEEN %(start_date)s AND %(end_date)s
  GROUP BY b1.slot
  UNION ALL
  SELECT b2.slot AS slot, toUInt64(b2.kzg_commitments_count) AS blob_count
  FROM `{network}`.libp2p_gossipsub_data_column_sidecar AS b2
  WHERE b2.slot_start_date_time BETWEEN %(start_date)s AND %(end_date)s
  GROUP BY b2.slot, b2.kzg_commitments_count
),
slot_blob AS (
  SELECT bc.slot AS slot, max(bc.blob_count) AS blob_count
  FROM blob_counts bc
  GROUP BY bc.slot
),
"""

    sampling_clause = f"AND rand() %% 100 < {sample_rate}" if sample_rate < 100 else ""

    query = f"""
WITH {blob_count_cte}base AS (
  SELECT slot, {diff_col} AS diff_ms, slot_start_date_time, meta_client_name
  FROM `{network}`.{table} FINAL
  WHERE slot_start_date_time BETWEEN %(start_date)s AND %(end_date)s
    AND {diff_col} < %(performance_threshold_ms)s
    AND {diff_col} IS NOT NULL
    AND meta_client_name != ''
    {sampling_clause}
),
proposer_meta AS (
  SELECT validator_index, {proposer_group_expr} AS proposer_group
  FROM (
    SELECT
      validator_index,
      coalesce(arrayElement(splitByChar(':', arrayFilter(x -> startsWith(x, 'cl:'), tags)[1]), 2), 'unknown') AS cl_client,
      coalesce(arrayElement(splitByChar(':', arrayFilter(x -> startsWith(x, 'el:'), tags)[1]), 2), 'unknown') AS el_client,
      IF(attributes['isClSupernode'] = 'true', 'supernode', 'regular') AS node_type,
      IF(arrayExists(x -> x = 'arch:arm', tags) OR lower(name) LIKE '%%%%arm%%%%', 'ARM', 'x86') AS architecture,
      coalesce(source, 'unknown') AS operator,
      coalesce(attributes['cloudRegion'], 'unknown') AS region,
      coalesce(attributes['cloud'], 'unknown') AS datacenter
    FROM `{network}`.dim_node FINAL
  ) vm
  WHERE 1=1 {proposer_filter_sql}
),
receiver_meta AS (
  SELECT DISTINCT meta_client_name, {receiver_group_expr} AS receiver_group
  FROM (
    SELECT DISTINCT
      meta_client_name,
      -- Use available beacon API metadata columns (corrected names)
      coalesce(meta_client_implementation,
        CASE
          WHEN meta_client_name LIKE '%%lighthouse%%' THEN 'lighthouse'
          WHEN meta_client_name LIKE '%%prysm%%' THEN 'prysm'
          WHEN meta_client_name LIKE '%%teku%%' THEN 'teku'
          WHEN meta_client_name LIKE '%%nimbus%%' THEN 'nimbus'
          WHEN meta_client_name LIKE '%%lodestar%%' THEN 'lodestar'
          WHEN meta_client_name LIKE '%%grandine%%' THEN 'grandine'
          ELSE 'unknown'
        END
      ) AS cl_client,
      'unknown' AS geo_continent,
      coalesce(meta_client_name, 'unknown') AS client_instance
    FROM `{network}`.{table} FINAL
    WHERE meta_client_name != ''
  ) vm
  WHERE 1=1 {receiver_filter_sql}
)
SELECT
  b.slot,
  CASE
    WHEN pm.proposer_group IS NOT NULL THEN pm.proposer_group
    ELSE 'unknown'
  END AS proposer_group,
  CASE
    WHEN rm.receiver_group IS NOT NULL THEN rm.receiver_group
    ELSE 'unknown'
  END AS receiver_group,
  b.diff_ms{', sb.blob_count' if enable_blob_bucketing else ''}
FROM base AS b
LEFT JOIN `{network}`.int_block_proposer_head AS iph ON b.slot = iph.slot
LEFT JOIN (
  SELECT DISTINCT slot
  FROM `{network}`.int_block_mev_head
  WHERE slot_start_date_time BETWEEN %(start_date)s AND %(end_date)s
) AS mev ON b.slot = mev.slot
LEFT JOIN proposer_meta AS pm ON iph.proposer_validator_index = pm.validator_index
LEFT JOIN receiver_meta AS rm ON b.meta_client_name = rm.meta_client_name{' LEFT JOIN slot_blob AS sb ON b.slot = sb.slot' if enable_blob_bucketing else ''}
WHERE 1=1{_build_mev_filter_clause(mev_filter, "if(mev.slot > 0, 'mev', 'non-mev')")}
ORDER BY b.slot_start_date_time DESC{f'\nLIMIT {max_records}' if max_records > 0 else ''}
"""
    return query


