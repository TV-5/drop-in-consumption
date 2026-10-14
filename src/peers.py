import pandas as pd

from .config import Config


def compute_peer_excess(drops: pd.DataFrame, master: pd.DataFrame,
                        cfg: Config) -> pd.DataFrame:
    """Adds peer_drop_pct, excess_drop_pct, peer_group_size, peer_level.

    Peer group: same dt_id, same tariff_category, sanctioned load within
    one band. Falls back to sub-division level (same tariff) when the group
    is smaller than cfg.min_peer_group_size. Widen the geography before
    widening the tariff.

    excess_drop_pct = drop_pct - peer_drop_pct

    This absorbs seasonality, local load shedding and festival effects with
    no weather data and no second year of history.
    """
    m = master[["meter_id", "dt_id", "subdivision_id", "tariff_category", "sanctioned_load_kw"]]
    df = drops.merge(m, on="meter_id", how="left")
    df["load_band"] = pd.cut(df["sanctioned_load_kw"], bins=cfg.load_band_edges_kw,
                             include_lowest=True).astype(str)

    valid = df["drop_pct"].notna()
    dt_keys = ["dt_id", "tariff_category", "load_band"]
    sd_keys = ["subdivision_id", "tariff_category"]

    def group_stats(keys):
        g = df[valid].groupby(keys)["drop_pct"].agg(peer_drop_pct="median", peer_group_size="count")
        return df[keys].merge(g, left_on=keys, right_index=True, how="left")[["peer_drop_pct", "peer_group_size"]]

    dt_stats = group_stats(dt_keys)
    sd_stats = group_stats(sd_keys)
    use_dt = dt_stats["peer_group_size"].fillna(0) >= cfg.min_peer_group_size

    df["peer_drop_pct"] = dt_stats["peer_drop_pct"].where(use_dt, sd_stats["peer_drop_pct"])
    df["peer_group_size"] = dt_stats["peer_group_size"].where(use_dt, sd_stats["peer_group_size"]).fillna(0).astype(int)
    df["peer_level"] = use_dt.map({True: "transformer", False: "sub-division"})
    df["excess_drop_pct"] = df["drop_pct"] - df["peer_drop_pct"]
    return df.drop(columns=["load_band"])
