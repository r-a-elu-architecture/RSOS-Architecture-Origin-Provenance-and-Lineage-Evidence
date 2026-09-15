from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import math
import re
import statistics
import traceback


ROOT = Path(
    r"C:\RSOS\RSOS_EXPERIMENTS_WF2_PROSPECTIVE_CROSS_SYSTEM_CONFIRMATION"
)

CONTROL = ROOT / "00_CONTROL"

FOUNDATION = (
    ROOT
    / "02_INPUTS"
    / "00_FOUNDATION_FROZEN"
)

DETECTOR = (
    FOUNDATION
    / "DETECTOR_PACKAGE"
)

PAIRSET = (
    FOUNDATION
    / "WF2_A_FROZEN_40_MATCHED_PAIRS.jsonl"
)

AUTHORITY = (
    FOUNDATION
    / "WF2_FOUNDATION_AUTHORITY_DECISION.json"
)

F2_SPEC = (
    DETECTOR
    / "F2_B_FROZEN_SPEC.json"
)

BR2_SPEC = (
    DETECTOR
    / "reference"
    / "F2_BR2_FROZEN_SPEC.json"
)

FEATURES = (
    DETECTOR
    / "reference"
    / "FROZEN_FEATURE_SETS.json"
)

README = (
    DETECTOR
    / "README.txt"
)

SCORER = (
    DETECTOR
    / "RUN_WF1_B_FRESH14_ADAPTER_R3.py"
)

F2_RUNNER = (
    DETECTOR
    / "RUN_F2_B.py"
)

SOURCE_BUILDER = (
    FOUNDATION
    / "SOURCE_RECORDS"
    / "01_BUILD_FROZEN_PROSPECTIVE_SET.py"
)

REPORT = (
    CONTROL
    / "PHASE0D_DESIGN_PREFLIGHT_REPORT.txt"
)

SUMMARY = (
    CONTROL
    / "PHASE0D_DESIGN_PREFLIGHT_SUMMARY.json"
)

ERRORLOG = (
    CONTROL
    / "PHASE0D_DESIGN_PREFLIGHT_ERROR.txt"
)


EXPECTED_PAIRSET_SHA = (
    "c43e54e5697fe6e35982ef4a559f909"
    "068f76e822edb7c6b172f536ef74090b5"
)


def sha256_file(path):

    h = hashlib.sha256()

    with path.open("rb") as f:

        while True:

            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def quantile(values, q):

    if not values:
        return None

    values = sorted(values)

    if len(values) == 1:
        return values[0]

    pos = (len(values) - 1) * q

    lower = math.floor(pos)
    upper = math.ceil(pos)

    if lower == upper:
        return values[lower]

    fraction = pos - lower

    return (
        values[lower] * (1 - fraction)
        + values[upper] * fraction
    )


def descriptive(values):

    if not values:

        return {
            "n": 0,
            "min": None,
            "q25": None,
            "median": None,
            "q75": None,
            "max": None,
            "mean": None
        }

    return {
        "n":
            len(values),

        "min":
            min(values),

        "q25":
            quantile(values, 0.25),

        "median":
            quantile(values, 0.50),

        "q75":
            quantile(values, 0.75),

        "max":
            max(values),

        "mean":
            statistics.mean(values)
    }


def extract_text(value):

    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, list):

        return "\n".join(
            extract_text(v)
            for v in value
        )

    if isinstance(value, dict):

        preferred = [
            "content",
            "text",
            "parts",
            "message",
            "value"
        ]

        pieces = []

        for key in preferred:

            if key in value:
                pieces.append(
                    extract_text(
                        value[key]
                    )
                )

        if pieces:

            return "\n".join(
                p
                for p in pieces
                if p
            )

        return "\n".join(
            extract_text(v)
            for v in value.values()
        )

    return str(value)


def get_role(message):

    if not isinstance(message, dict):
        return "UNKNOWN"

    for key in (
        "role",
        "author_role",
        "speaker",
        "author"
    ):

        value = message.get(key)

        if isinstance(value, str):
            return value.upper()

        if isinstance(value, dict):

            role = value.get("role")

            if isinstance(role, str):
                return role.upper()

    return "UNKNOWN"


def read_text(path):

    if not path.exists():
        return ""

    return path.read_text(
        encoding="utf-8-sig",
        errors="ignore"
    )


def code_excerpt(path, patterns, max_lines=300):

    if not path.exists():
        return [
            f"NOT FOUND: {path}"
        ]

    text = read_text(path)

    source_lines = text.splitlines()

    selected = []

    regex = re.compile(
        "|".join(
            f"(?:{p})"
            for p in patterns
        ),
        flags=re.I
    )

    for number, line in enumerate(
        source_lines,
        start=1
    ):

        if regex.search(line):

            selected.append(
                f"{number:05d}: {line}"
            )

        if len(selected) >= max_lines:
            break

    return selected


try:

    # ========================================================
    # 1. VERIFY FROZEN SOURCE SET IDENTITY
    # ========================================================

    pairset_sha = sha256_file(
        PAIRSET
    )

    if pairset_sha != EXPECTED_PAIRSET_SHA:

        raise RuntimeError(
            "Frozen prospective source-set SHA mismatch."
        )


    # ========================================================
    # 2. LOAD 80 FROZEN SOURCE ROWS
    # ========================================================

    rows = []

    with PAIRSET.open(
        "r",
        encoding="utf-8"
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1
        ):

            if not line.strip():
                continue

            obj = json.loads(line)

            obj["_line_number"] = line_number

            rows.append(obj)


    if len(rows) != 80:

        raise RuntimeError(
            f"Expected 80 rows; found {len(rows)}."
        )


    # ========================================================
    # 3. PAIR / LABEL INTEGRITY
    # ========================================================

    label_counts = Counter()

    pair_members = defaultdict(list)

    model_slug_counts = Counter()

    for row in rows:

        label = str(
            row.get(
                "LABEL",
                "MISSING"
            )
        )

        pair_index = str(
            row.get(
                "PAIR_INDEX",
                "MISSING"
            )
        )

        model_slug = str(
            row.get(
                "DEFAULT_MODEL_SLUG",
                "MISSING"
            )
        )

        label_counts[label] += 1

        pair_members[pair_index].append(
            label
        )

        model_slug_counts[
            model_slug
        ] += 1


    complete_pairs = 0
    malformed_pairs = {}

    for pair_index, labels in pair_members.items():

        if sorted(labels) == sorted(
            [
                "LINEAGE_POS",
                "MATCHED_NEG"
            ]
        ):

            complete_pairs += 1

        else:

            malformed_pairs[
                pair_index
            ] = labels


    # ========================================================
    # 4. SOURCE MESSAGE GEOMETRY
    # ========================================================

    overall_message_counts = []
    overall_character_counts = []
    overall_token_estimates = []

    message_counts_by_label = defaultdict(list)
    character_counts_by_label = defaultdict(list)
    token_estimates_by_label = defaultdict(list)

    role_counts = Counter()
    final_role_counts = Counter()

    empty_message_rows = []

    per_row_stats = []


    for row in rows:

        label = str(
            row.get("LABEL")
        )

        messages = row.get(
            "SOURCE_MESSAGES"
        )

        if not isinstance(messages, list):

            messages = []


        msg_count = len(messages)

        char_count = 0

        row_roles = []

        for message in messages:

            role = get_role(
                message
            )

            role_counts[role] += 1

            row_roles.append(
                role
            )

            text = extract_text(
                message
            )

            char_count += len(
                text
            )


        if row_roles:

            final_role_counts[
                row_roles[-1]
            ] += 1


        # Rough pre-API planning estimate only.
        token_estimate = math.ceil(
            char_count / 4
        )


        overall_message_counts.append(
            msg_count
        )

        overall_character_counts.append(
            char_count
        )

        overall_token_estimates.append(
            token_estimate
        )

        message_counts_by_label[
            label
        ].append(
            msg_count
        )

        character_counts_by_label[
            label
        ].append(
            char_count
        )

        token_estimates_by_label[
            label
        ].append(
            token_estimate
        )


        if msg_count == 0:

            empty_message_rows.append(
                row["_line_number"]
            )


        per_row_stats.append({
            "line_number":
                row["_line_number"],

            "pair_index":
                row.get("PAIR_INDEX"),

            "label":
                label,

            "model_slug":
                row.get(
                    "DEFAULT_MODEL_SLUG"
                ),

            "message_count":
                msg_count,

            "characters":
                char_count,

            "estimated_tokens_char_div_4":
                token_estimate,

            "last_role":
                (
                    row_roles[-1]
                    if row_roles
                    else None
                )
        })


    # ========================================================
    # 5. DETECTOR / FEATURE SPECIFICATION
    # ========================================================

    authority_text = read_text(
        AUTHORITY
    )

    f2_spec_text = read_text(
        F2_SPEC
    )

    br2_spec_text = read_text(
        BR2_SPEC
    )

    feature_text = read_text(
        FEATURES
    )

    readme_text = read_text(
        README
    )


    # ========================================================
    # 6. RUNNER / ADAPTER EXECUTION REQUIREMENTS
    # ========================================================

    scoring_patterns = [
        r"^def\s+",
        r"joblib",
        r"decision_function",
        r"predict_proba",
        r"feature",
        r"window",
        r"messages",
        r"conversation",
        r"jsonl",
        r"assistant",
        r"user",
        r"threshold",
        r"margin",
        r"JOINT",
        r"OPERATOR",
        r"SURFACE",
        r"R1"
    ]

    scorer_excerpt = code_excerpt(
        SCORER,
        scoring_patterns,
        max_lines=350
    )

    f2_runner_excerpt = code_excerpt(
        F2_RUNNER,
        scoring_patterns,
        max_lines=250
    )

    builder_excerpt = code_excerpt(
        SOURCE_BUILDER,
        [
            r"^def\s+",
            r"PAIR",
            r"LABEL",
            r"SOURCE_MESSAGES",
            r"SHA",
            r"rank",
            r"match",
            r"sample",
            r"seed",
            r"40",
            r"80"
        ],
        max_lines=250
    )


    # ========================================================
    # 7. BUILD MACHINE SUMMARY
    # ========================================================

    source_geometry = {

        "rows":
            len(rows),

        "pair_count":
            len(pair_members),

        "complete_pairs":
            complete_pairs,

        "malformed_pairs":
            malformed_pairs,

        "label_counts":
            dict(label_counts),

        "source_model_slug_counts":
            dict(model_slug_counts),

        "message_count_overall":
            descriptive(
                overall_message_counts
            ),

        "character_count_overall":
            descriptive(
                overall_character_counts
            ),

        "rough_token_estimate_overall":
            descriptive(
                overall_token_estimates
            ),

        "message_count_by_label": {
            label:
                descriptive(values)

            for label, values
            in message_counts_by_label.items()
        },

        "rough_tokens_by_label": {
            label:
                descriptive(values)

            for label, values
            in token_estimates_by_label.items()
        },

        "message_role_counts":
            dict(role_counts),

        "final_role_counts":
            dict(final_role_counts),

        "empty_message_rows":
            empty_message_rows
    }


    summary = {

        "phase":
            "WF2_PHASE0D_DESIGN_PREFLIGHT",

        "pairset_sha256":
            pairset_sha,

        "source_geometry":
            source_geometry,

        "per_row_stats":
            per_row_stats,

        "scientific_observations_created":
            False,

        "api_calls_performed":
            False,

        "prospective_generation_performed":
            False,

        "detector_training_performed":
            False,

        "threshold_tuning_performed":
            False,

        "boundary_1_reached":
            False
    }


    SUMMARY.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=True
        ),
        encoding="utf-8"
    )


    # ========================================================
    # 8. BUILD HUMAN-READABLE REPORT
    # ========================================================

    out = []

    out.extend([
        "=" * 90,
        "WF2 PHASE 0D - GENERATION / SCORING DESIGN PREFLIGHT",
        "=" * 90,
        "",
        f"WF2 ROOT: {ROOT}",
        "",
        f"PAIRSET SHA256: {pairset_sha}",
        "",
        "SCIENTIFIC OBSERVATIONS CREATED: NO",
        "API CALLS PERFORMED: NO",
        "PROSPECTIVE GENERATION PERFORMED: NO",
        "DETECTOR TRAINING PERFORMED: NO",
        "THRESHOLD TUNING PERFORMED: NO",
        "BOUNDARY 1 REACHED: NO",
        ""
    ])


    # --------------------------------------------------------
    # SOURCE INTEGRITY
    # --------------------------------------------------------

    out.extend([
        "=" * 90,
        "A. FROZEN SOURCE-SET INTEGRITY",
        "=" * 90,
        "",
        f"Rows: {len(rows)}",
        f"Pairs: {len(pair_members)}",
        f"Complete matched pairs: {complete_pairs}",
        f"Malformed pairs: {len(malformed_pairs)}",
        f"Labels: {dict(label_counts)}",
        "",
        "Historical source model slugs:"
    ])

    for name, count in sorted(
        model_slug_counts.items(),
        key=lambda x: (-x[1], x[0])
    ):

        out.append(
            f"  {name}: {count}"
        )

    out.append("")


    # --------------------------------------------------------
    # SOURCE GEOMETRY
    # --------------------------------------------------------

    out.extend([
        "=" * 90,
        "B. SOURCE CONTEXT GEOMETRY",
        "=" * 90,
        ""
    ])


    def add_stats(title, stats):

        out.append(title)

        for key in (
            "n",
            "min",
            "q25",
            "median",
            "q75",
            "max",
            "mean"
        ):

            out.append(
                f"  {key}: {stats.get(key)}"
            )

        out.append("")


    add_stats(
        "Messages per source context:",
        descriptive(
            overall_message_counts
        )
    )

    add_stats(
        "Characters per source context:",
        descriptive(
            overall_character_counts
        )
    )

    add_stats(
        "Rough token estimate per context (characters / 4):",
        descriptive(
            overall_token_estimates
        )
    )


    for label in sorted(
        message_counts_by_label
    ):

        add_stats(
            f"Messages - {label}:",
            descriptive(
                message_counts_by_label[
                    label
                ]
            )
        )

        add_stats(
            f"Rough tokens - {label}:",
            descriptive(
                token_estimates_by_label[
                    label
                ]
            )
        )


    out.append(
        f"Message role counts: {dict(role_counts)}"
    )

    out.append(
        f"Final message roles: {dict(final_role_counts)}"
    )

    out.append(
        f"Rows with zero messages: {empty_message_rows}"
    )

    out.append("")


    # --------------------------------------------------------
    # PER-ROW DISTRIBUTION
    # --------------------------------------------------------

    out.extend([
        "=" * 90,
        "C. PER-ROW SOURCE GEOMETRY",
        "=" * 90,
        "",
        "line | pair | label | model | messages | chars | rough_tokens | last_role"
    ])


    for row in sorted(
        per_row_stats,
        key=lambda x: (
            str(x["pair_index"]),
            x["label"]
        )
    ):

        out.append(
            f"{row['line_number']} | "
            f"{row['pair_index']} | "
            f"{row['label']} | "
            f"{row['model_slug']} | "
            f"{row['message_count']} | "
            f"{row['characters']} | "
            f"{row['estimated_tokens_char_div_4']} | "
            f"{row['last_role']}"
        )


    out.append("")


    # --------------------------------------------------------
    # AUTHORITY DECISION
    # --------------------------------------------------------

    out.extend([
        "=" * 90,
        "D. CURRENT FOUNDATION AUTHORITY DECISION",
        "=" * 90,
        "",
        authority_text,
        ""
    ])


    # --------------------------------------------------------
    # DETECTOR SPECS
    # --------------------------------------------------------

    out.extend([
        "=" * 90,
        "E. F2_B_FROZEN_SPEC.json",
        "=" * 90,
        "",
        f2_spec_text,
        "",
        "=" * 90,
        "F. F2_BR2_FROZEN_SPEC.json",
        "=" * 90,
        "",
        br2_spec_text,
        "",
        "=" * 90,
        "G. FROZEN_FEATURE_SETS.json",
        "=" * 90,
        "",
        feature_text,
        "",
        "=" * 90,
        "H. DETECTOR PACKAGE README",
        "=" * 90,
        "",
        readme_text,
        ""
    ])


    # --------------------------------------------------------
    # EXECUTION CONTRACT EXCERPTS
    # --------------------------------------------------------

    out.extend([
        "=" * 90,
        "I. FINAL WF1-B ADAPTER - RELEVANT CODE LINES",
        "=" * 90,
        ""
    ])

    out.extend(
        scorer_excerpt
    )

    out.extend([
        "",
        "=" * 90,
        "J. F2-B RUNNER - RELEVANT CODE LINES",
        "=" * 90,
        ""
    ])

    out.extend(
        f2_runner_excerpt
    )

    out.extend([
        "",
        "=" * 90,
        "K. ORIGINAL V13 SOURCE-SET BUILDER - RELEVANT CODE LINES",
        "=" * 90,
        ""
    ])

    out.extend(
        builder_excerpt
    )


    # --------------------------------------------------------
    # FINAL DESIGN GATE
    # --------------------------------------------------------

    out.extend([
        "",
        "=" * 90,
        "L. DESIGN GATE",
        "=" * 90,
        "",
        "This preflight does NOT choose a prospective generation length.",
        "It exposes the frozen source geometry and detector execution contract",
        "so the unified WF2-A/B/C protocol can be fixed before observation.",
        "",
        "No API output may be generated before Boundary 1 is frozen."
    ])


    REPORT.write_text(
        "\n".join(out),
        encoding="utf-8"
    )


    if ERRORLOG.exists():

        ERRORLOG.unlink()


    print(
        REPORT.read_text(
            encoding="utf-8"
        )
    )


except Exception:

    error = traceback.format_exc()

    ERRORLOG.write_text(
        error,
        encoding="utf-8"
    )

    print("")
    print("=" * 90)
    print("WF2 PHASE 0D ERROR")
    print("=" * 90)
    print("")
    print(error)
    print("")
    print(f"ERROR LOG: {ERRORLOG}")

