"""Render the committed, hash-bound H7 supplementary forest plot."""

from src.h7_feature_transfer_figure import render_production


def main() -> None:
    hashes = render_production()
    print(f"Rendered H7 feature-transfer figure: {hashes}")


if __name__ == "__main__":
    main()
