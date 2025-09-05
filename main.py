# main.py
import sys
from data import preprocessing_data
from ila import ILA
from model import ILAModel
from utils import format_rule

# ================== MAIN ==================


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Train: python ila_main.py train <training_file> [model_file]")
        print("  Predict: python ila_main.py predict <model_file> <test_file>")
        print("  Legacy: python ila_main.py <data_file>")
        print("\nSupported formats: .csv, .xlsx, .xls")
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == "train":
        if len(sys.argv) < 3:
            print(
                "Usage: python ila_main.py train <training_file> [model_file]")
            sys.exit(1)

        training_file = sys.argv[2]
        model_file = sys.argv[3] if len(sys.argv) > 3 else "ila_model.pkl"

        print(f"Training ILA model on {training_file}...")
        model = ILAModel()
        model.fit(training_file)

        print(f"\n== Learned Rules ==")
        rules = model.get_rules()
        for i, rule in enumerate(rules, 1):
            print(f"Rule {i}: {rule}")

        model.save_model(model_file)
        print(f"\nModel saved to {model_file}")
        print(f"Total rules learned: {len(rules)}")

    elif mode == "predict":
        if len(sys.argv) < 4:
            print("Usage: python ila_main.py predict <model_file> <test_file>")
            sys.exit(1)

        model_file = sys.argv[2]
        test_file = sys.argv[3]

        print(f"Loading model from {model_file}...")
        model = ILAModel()
        model.load_model(model_file)

        print(f"Making predictions on {test_file}...")
        predictions, accuracy = model.predict_with_accuracy(test_file)

        print(f"\n== Predictions ==")
        for i, pred in enumerate(predictions, 1):
            print(f"Instance {i}: {pred}")

        if accuracy is not None:
            print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        else:
            print("\nNo true labels available for accuracy calculation")

    else:
        # Legacy mode - original behavior
        file_path = sys.argv[1]
        enc = preprocessing_data(file_path)

        # Sinh luật
        rules = ILA(enc)

        # In luật
        print("\n== Tập luật ILA ==")
        for i, r in enumerate(rules, 1):
            print(f"Rule {i}: {format_rule(r, enc)}")


if __name__ == "__main__":
    main()
