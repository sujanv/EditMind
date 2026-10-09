#!/usr/bin/env python3
"""Interactive demo of knowledge editing and causal tracing."""

from editmind.models.model_wrapper import UnifiedModelWrapper
from editmind.models.causal_tracer import CausalTracer
from editmind.editors.rome import ROMEEditor
from editmind.editors.memit import MEMITEditor
from editmind.core.types import EditRequest
from editmind.visualization.causal_plots import render_ascii_causal_heatmap


def main():
    print("=" * 70)
    print("          EditMind: Knowledge Editing in LLMs Demo")
    print("=" * 70)

    # 1. Initialize Model
    print("\n[1] Initializing Unified Causal Model...")
    wrapper = UnifiedModelWrapper.create_toy_model(hidden_dim=128, num_layers=6)

    prompt = "The Eiffel Tower is in"
    subject = "Eiffel Tower"
    target_new = "Rome"
    ground_truth = "Paris"

    print(f"Target Prompt: '{prompt}'")
    print(f"New Target  : '{target_new}' (Ground truth: '{ground_truth}')")

    # 2. Causal Mediation Analysis
    print("\n[2] Running Causal Mediation Analysis to locate factual memory...")
    tracer = CausalTracer(wrapper, noise_std=0.15)
    trace_res = tracer.trace(prompt=prompt, subject=subject, target=ground_truth)
    print(render_ascii_causal_heatmap(trace_res))

    best_l, best_t, tok = trace_res.find_critical_layer_and_token()
    print(f"=> Causal attribution confirms factual storage in Layer {best_l} at token '{tok}'.")

    # 3. Apply ROME Edit
    target_layer = 2
    print(f"\n[3] Applying ROME (Rank-One Model Editing) at Layer {target_layer}...")
    editor = ROMEEditor(wrapper, config={"target_layer": target_layer, "v_num_grad_steps": 30, "v_lr": 0.5})
    req = EditRequest(
        prompt=prompt,
        target_new=target_new,
        ground_truth=ground_truth,
        subject=subject,
    )
    result = editor.edit(req)

    print(f"Edit Outcome         : {'SUCCESS' if result.success else 'FAILED'}")
    print(f"Execution Latency    : {result.execution_time_sec * 1000:.2f} ms")
    print(f"P({target_new}) Pre  : {result.pre_edit_target_prob:.4f}")
    print(f"P({target_new}) Post : {result.post_edit_target_prob:.4f}")
    print(f"Weight Delta ||W||   : {result.delta_weight_norm:.4f}")

    # 4. Verification
    print("\n[4] Verifying generated text post-edit...")
    gen_text = wrapper.generate(prompt, max_new_tokens=3)
    print(f"Prompt continuation  : '{prompt} {gen_text}'")

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
