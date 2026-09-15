from pathlib import Path
from collections import defaultdict
import importlib.util
import json

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


ROOT = Path(
    r"C:\RSOS\RSOS_EXPERIMENTS_WF2_PROSPECTIVE_CROSS_SYSTEM_CONFIRMATION"
)

STAGE = (
    ROOT
    / "02_INPUTS"
    / "00_FOUNDATION_FROZEN"
    / "WF1_A_ACTIVE_AUTHORITY"
)

RAW = (
    ROOT
    / "04_RAW"
    / "ACTIVE_CAUSAL"
)

OUT = (
    ROOT
    / "05_OUTPUTS"
    / "ACTIVE_CAUSAL"
)

ADJ = (
    ROOT
    / "07_ADJUDICATION"
)


ORIGINAL = (
    STAGE
    / "RUN_WF1_A_V12_R_CORRECTIVE_CAUSAL_TOPOLOGY_R1.py"
)

HIST = (
    STAGE
    / "07_SEED_LEVEL_GRAPH_VECTOR.csv"
)


def load_original():

    spec = importlib.util.spec_from_file_location(
        "wf1a_original",
        ORIGINAL
    )

    m = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(m)

    return m


def one_sided_signflip(
    m,
    values,
    seed
):

    return float(
        m.signflip(
            np.asarray(
                values,
                float
            ),
            seed
        )
    )


def main():

    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    m = load_original()

    hist = pd.read_csv(
        HIST
    )


    means,scales,precision,shrink = (
        m.fit_distance_model(
            hist
        )
    )


    raw_files = list(
        RAW.glob("*.json")
    )


    rows = []


    for path in raw_files:

        rec = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )


        if rec.get("status") != "COMPLETE":
            continue


        seed = int(
            rec["seed"]
        )

        condition = rec[
            "condition"
        ]


        prompt,payload,meta = (
            m.build_prompt(
                seed,
                condition
            )
        )


        parsed = m.parse_json_object(
            rec["text"]
        )


        valid,reason = m.validate_output(
            parsed,
            payload
        )


        if not valid:
            continue


        feature = m.extract_features(
            seed,
            condition,
            payload,
            meta,
            parsed
        )


        feature.update({
            "PROVIDER":
                rec["provider"],

            "MODEL":
                rec["model"],

            "CELL_ID":
                rec["cell_id"]
        })


        rows.append(
            feature
        )


    sdf = pd.DataFrame(
        rows
    )


    sdf.to_csv(
        OUT
        /
        "WF2_ACTIVE_CAUSAL_FEATURES.csv",
        index=False
    )


    providers = sorted(
        sdf.PROVIDER.unique()
    )


    pure_delete = [
        f"E{i}_DELETE"
        for i in [
            0,1,2,3,4,5,6,8,9
        ]
    ]

    pure_reverse = [
        f"E{i}_REVERSE"
        for i in [
            0,1,2,3,4,5,6,8
        ]
    ]

    specificity = [
        "R_DEGREE_MATCHED_REWIRE",
        "X_COMPLEXITY_MATCHED",
        "Y_TOPOLOGY_MATCHED"
    ]


    historical_train = hist[
        hist.SEED.isin(
            m.DEV
        )
        &
        hist.CONDITION.isin({
            "A_AUTHENTIC",
            "I_ISOMORPHIC_RELABEL",
            "F_SURFACE_FORMAT_CONTROL",
            *specificity
        })
    ].copy()


    historical_train[
        "Y"
    ] = (
        historical_train.CONDITION.isin({
            "A_AUTHENTIC",
            "I_ISOMORPHIC_RELABEL",
            "F_SURFACE_FORMAT_CONTROL"
        })
    ).astype(int)


    classifier = Pipeline([
        (
            "scale",
            StandardScaler()
        ),
        (
            "lr",
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                random_state=m.MASTER_SEED
            )
        )
    ])


    classifier.fit(
        historical_train[
            m.BEHAVIOR_FEATURES
        ],
        historical_train.Y
    )


    results = {}


    for provider in providers:

        pdf = sdf[
            sdf.PROVIDER
            ==
            provider
        ].copy()


        required = (
            {
                "A_AUTHENTIC",
                "I_ISOMORPHIC_RELABEL",
                "F_SURFACE_FORMAT_CONTROL"
            }
            |
            set(pure_delete)
            |
            set(pure_reverse)
            |
            set(specificity)
        )


        seed_sets = []

        for condition in required:

            seed_sets.append(
                set(
                    pdf[
                        pdf.CONDITION
                        ==
                        condition
                    ].SEED
                )
            )


        complete_seeds = sorted(
            set.intersection(
                *seed_sets
            )
            if seed_sets
            else set()
        )


        distances = []


        for seed in complete_seeds:

            a = pdf[
                (pdf.SEED==seed)
                &
                (
                    pdf.CONDITION
                    ==
                    "A_AUTHENTIC"
                )
            ].iloc[0]


            for condition in required:

                b = pdf[
                    (pdf.SEED==seed)
                    &
                    (
                        pdf.CONDITION
                        ==
                        condition
                    )
                ].iloc[0]


                mah,eu = m.distances(
                    a,
                    b,
                    scales,
                    precision
                )


                distances.append({
                    "SEED":
                        seed,

                    "CONDITION":
                        condition,

                    "MAHALANOBIS":
                        mah,

                    "STD_EUCLIDEAN":
                        eu
                })


        ddf = pd.DataFrame(
            distances
        )


        nuisance = {}


        for seed in complete_seeds:

            vals = []

            for c in [
                "I_ISOMORPHIC_RELABEL",
                "F_SURFACE_FORMAT_CONTROL"
            ]:

                vals.append(
                    float(
                        ddf[
                            (ddf.SEED==seed)
                            &
                            (
                                ddf.CONDITION
                                ==
                                c
                            )
                        ].MAHALANOBIS.iloc[0]
                    )
                )


            nuisance[
                seed
            ] = float(
                np.mean(
                    vals
                )
            )


        def family_test(
            conditions,
            seed_offset
        ):

            diffs = []


            for seed in complete_seeds:

                vals = [
                    float(
                        ddf[
                            (ddf.SEED==seed)
                            &
                            (
                                ddf.CONDITION
                                ==
                                c
                            )
                        ].MAHALANOBIS.iloc[0]
                    )
                    for c in conditions
                ]


                diffs.append(
                    float(
                        np.mean(
                            vals
                        )
                    )
                    -
                    nuisance[
                        seed
                    ]
                )


            return {
                "N":
                    len(diffs),

                "EXCESS":
                    float(
                        np.mean(
                            diffs
                        )
                    ),

                "P":
                    one_sided_signflip(
                        m,
                        diffs,
                        m.MASTER_SEED
                        +
                        seed_offset
                    )
            }


        delete_test = family_test(
            pure_delete,
            9101
        )

        reverse_test = family_test(
            pure_reverse,
            9102
        )


        spec_rows = []


        for index,c in enumerate(
            specificity
        ):

            diffs = []

            for seed in complete_seeds:

                val = float(
                    ddf[
                        (ddf.SEED==seed)
                        &
                        (
                            ddf.CONDITION
                            ==
                            c
                        )
                    ].MAHALANOBIS.iloc[0]
                )


                diffs.append(
                    val
                    -
                    nuisance[
                        seed
                    ]
                )


            spec_rows.append({
                "CONDITION":
                    c,

                "EXCESS":
                    float(
                        np.mean(
                            diffs
                        )
                    ),

                "P":
                    one_sided_signflip(
                        m,
                        diffs,
                        m.MASTER_SEED
                        +
                        9200
                        +
                        index
                    )
            })


        adjusted = m.holm_adjust(
            [
                x["P"]
                for x in spec_rows
            ]
        )


        for row,padj in zip(
            spec_rows,
            adjusted
        ):
            row["HOLM_P"] = float(
                padj
            )


        hdf = pdf[
            pdf.CONDITION.isin({
                "A_AUTHENTIC",
                "I_ISOMORPHIC_RELABEL",
                "F_SURFACE_FORMAT_CONTROL",
                *specificity
            })
        ].copy()


        hdf["Y"] = (
            hdf.CONDITION.isin({
                "A_AUTHENTIC",
                "I_ISOMORPHIC_RELABEL",
                "F_SURFACE_FORMAT_CONTROL"
            })
        ).astype(int)


        prob = classifier.predict_proba(
            hdf[
                m.BEHAVIOR_FEATURES
            ]
        )[:,1]


        hard_auc = float(
            roc_auc_score(
                hdf.Y,
                prob
            )
        )


        provider_pass = (
            hard_auc >= 0.85

            and
            delete_test[
                "EXCESS"
            ] > 0

            and
            delete_test[
                "P"
            ] < 0.01

            and
            reverse_test[
                "EXCESS"
            ] > 0

            and
            reverse_test[
                "P"
            ] < 0.01

            and
            all(
                x["EXCESS"] > 0
                and
                x["HOLM_P"] < 0.01

                for x
                in spec_rows
            )
        )


        results[
            provider
        ] = {

            "complete_seeds":
                len(
                    complete_seeds
                ),

            "hard_control_auc":
                hard_auc,

            "pure9_delete":
                delete_test,

            "pure8_reverse":
                reverse_test,

            "specificity":
                spec_rows,

            "provider_strong_gate":
                bool(
                    provider_pass
                )
        }


    passes = sum(
        int(
            x[
                "provider_strong_gate"
            ]
        )
        for x in results.values()
    )


    if passes >= 2:

        finding = "POSITIVE"

    elif passes == 1:

        finding = "MIXED"

    else:

        finding = "NULL_INCONCLUSIVE"


    adjudication = {

        "EXECUTION":
            (
                "COMPLETED"
                if len(results) == 3
                else "INCOMPLETE"
            ),

        "SCIENTIFIC_VALIDITY":
            (
                "VALID"
                if len(results) == 3
                else "UNRESOLVED"
            ),

        "FINDING":
            finding,

        "AUTHORITY":
            (
                "FINAL_AUTHORITY"
                if len(results) == 3
                else "UNRESOLVED_AUTHORITY"
            ),

        "PUBLICATION":
            (
                "ELIGIBLE_FINAL"
                if len(results) == 3
                else "DO_NOT_CITE_AS_RESULT"
            ),

        "PROVIDER_RESULTS":
            results,

        "NOTE":
            (
                "Active causal/topology replication is adjudicated "
                "separately from passive WF2. It cannot rescue a "
                "negative passive result and vice versa."
            )
    }


    (
        ADJ
        /
        "WF2_ACTIVE_CAUSAL_R1_ADJUDICATION.json"
    ).write_text(
        json.dumps(
            adjudication,
            indent=2,
            sort_keys=True
        ),
        encoding="utf-8"
    )


    print(
        json.dumps(
            adjudication,
            indent=2
        )
    )


if __name__ == "__main__":
    main()
