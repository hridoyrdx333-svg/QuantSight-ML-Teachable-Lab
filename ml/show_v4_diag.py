import json

with open(
    "ml/reports/v4_diagnostics.json",
    "r",
    encoding="utf-8",
) as f:
    d = json.load(f)

print("=== TOP 15 FEATURES ===")

for i, x in enumerate(
    d["random_forest_feature_importance"][:15],
    start=1,
):
    print(
        f"{i:02d}. "
        f"{x['feature']}: "
        f"{x['importance']:.6f}"
    )

print()

print("=== PER SYMBOL RANDOM FOREST ===")

for symbol in d["per_symbol"]:
    result = d["per_symbol"][symbol]["random_forest"]

    print()
    print(symbol)

    for key, value in result.items():
        print(f"  {key}: {value}")