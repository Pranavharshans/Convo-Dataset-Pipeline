from convo_ds.config import default_config


def test_default_config_has_50k_conversations() -> None:
    config = default_config()
    assert sum(bucket.count for bucket in config.buckets) == 50000


def test_default_config_has_expected_stage3_hours() -> None:
    config = default_config()
    seconds = sum(bucket.count * bucket.duration_avg_sec for bucket in config.buckets)
    assert round(seconds / 3600) == 449
