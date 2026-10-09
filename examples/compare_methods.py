"""Compare 5 Knowledge Editing Approaches Side-by-Side."""

from editmind.models import UnifiedModelWrapper
from editmind.data import load_counterfact_dataset
from editmind.evaluation import MethodComparator

def main():
    print("Loading model and dataset...")
    wrapper = UnifiedModelWrapper.create_toy_model(hidden_dim=128, num_layers=6)
    dataset = load_counterfact_dataset(sample_count=3)

    comparator = MethodComparator(
        model_wrapper=wrapper,
        methods=["rome", "memit", "mend", "grace", "ike", "ft_l"]
    )

    print("\nRunning side-by-side benchmark across methods...")
    results = comparator.compare_on_dataset(dataset)

    print("\n" + comparator.generate_markdown_table(results))

if __name__ == "__main__":
    main()
