"""Mock office and network hierarchy that the synthetic population is generated under.

Office:  Circle (SE) -> Division (EE) -> Sub-division (AE)
Network: 33/11 kV substation -> 11 kV feeder -> distribution transformer (DT) -> meter
"""
import numpy as np
import pandas as pd

CIRCLE = {"id": "CIR-JPR", "name": "Jaipur City Circle",
          "se_name": "SE Jaipur City Circle", "se_email": "se.jaipur.city@example.com"}

DIVISIONS = {
    "DIV-JPR-N": {"name": "Jaipur North Division", "ee_name": "EE Jaipur North"},
    "DIV-JPR-S": {"name": "Jaipur South Division", "ee_name": "EE Jaipur South"},
}

# sub-division -> division, name, AE, installed smart meters, and its network: {substation: {feeder: number of DTs}}
SUBDIVISIONS = {
    "SD-JPR-07": {"division": "DIV-JPR-N", "name": "Vidyadhar Nagar", "ae_name": "AE Vidyadhar Nagar", "meters": 540,
                  "network": {"SS-33/11-Vidyadhar Nagar": {"F-11-VDN-01": 3, "F-11-VDN-02": 3},
                              "SS-33/11-Sikar Road": {"F-11-SKR-01": 4}}},
    "SD-JPR-08": {"division": "DIV-JPR-N", "name": "Shastri Nagar", "ae_name": "AE Shastri Nagar", "meters": 420,
                  "network": {"SS-33/11-Shastri Nagar": {"F-11-SHN-01": 3, "F-11-SHN-02": 2, "F-11-SHN-03": 3}}},
    "SD-JPR-11": {"division": "DIV-JPR-S", "name": "Malviya Nagar", "ae_name": "AE Malviya Nagar", "meters": 610,
                  "network": {"SS-33/11-Malviya Nagar": {"F-11-MVN-01": 4, "F-11-MVN-02": 3},
                              "SS-33/11-Jawahar Circle": {"F-11-JWC-01": 3}}},
    "SD-JPR-12": {"division": "DIV-JPR-S", "name": "Sanganer", "ae_name": "AE Sanganer", "meters": 480,
                  "network": {"SS-33/11-Sanganer": {"F-11-SNG-01": 3, "F-11-SNG-02": 3},
                              "SS-33/11-Pratap Nagar": {"F-11-PTN-01": 2, "F-11-PTN-02": 2}}},
}


def label(sd_id: str) -> str:
    sd = SUBDIVISIONS[sd_id]
    return f"{sd_id} · {sd['name']} ({DIVISIONS[sd['division']]['name']}, {CIRCLE['name']})"


def network(sd_id: str) -> pd.DataFrame:
    """One row per DT: dt_id, feeder_id, substation_id, meters (installed smart meters on that DT)."""
    code = sd_id.removeprefix("SD-")
    rows, k = [], 0
    for ss, feeders in SUBDIVISIONS[sd_id]["network"].items():
        for feeder, n_dts in feeders.items():
            for _ in range(n_dts):
                k += 1
                rows.append({"dt_id": f"DT-{code}-{k:03d}", "feeder_id": feeder, "substation_id": ss})
    net = pd.DataFrame(rows)
    # Fixed split of the sub-division's meters across its DTs: part of the area's description, not of the seed.
    rng = np.random.default_rng(int(sd_id.rsplit("-", 1)[1]))
    net["meters"] = rng.multinomial(SUBDIVISIONS[sd_id]["meters"], rng.dirichlet(np.full(len(net), 8.0)))
    return net


def officer(sd_id: str) -> pd.DataFrame:
    """The SE who receives this sub-division's inspection list (officers contract)."""
    return pd.DataFrame({"subdivision_id": [sd_id], "se_name": [CIRCLE["se_name"]], "se_email": [CIRCLE["se_email"]]})
