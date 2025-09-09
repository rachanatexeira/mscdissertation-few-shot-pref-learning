import torch

# Load model state dicts directly
state1 = torch.load("results/fetchreach_run/best_model.pt")
state2 = torch.load("results/fetchreach_run/final_model.pt")

def compare_state_dicts(s1, s2):
    keys1 = set(s1.keys())
    keys2 = set(s2.keys())

    if keys1 != keys2:
        print("❌ Keys differ between models:")
        print("Only in best_model.pt:", keys1 - keys2)
        print("Only in final_model.pt:", keys2 - keys1)
        return

    all_match = True
    for k in sorted(s1.keys()):
        val1 = s1[k]
        val2 = s2[k]

        try:
            if isinstance(val1, torch.Tensor) and isinstance(val2, torch.Tensor):
                if not torch.allclose(val1, val2, atol=1e-6):
                    print(f"❌ Tensor mismatch at key: {k} — Shapes: {val1.shape} vs {val2.shape}")
                    all_match = False
            else:
                if val1 != val2:
                    print(f"❌ Value mismatch at key: {k} — {val1} vs {val2}")
                    all_match = False
        except Exception as e:
            print(f"⚠️ Exception comparing key {k}: {e}")
            all_match = False

    if all_match:
        print("✅ The two models are identical.")
    else:
        print("⚠️ Some values differ between the models.")

# ✅ Call the function
compare_state_dicts(state1, state2)