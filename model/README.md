# OutRo model

The loader uses the original Transformers Qwen2.5-Omni model for baseline inference.
With `use_outro=True`, it loads `modeling_qwen2_5_omni.py`, which applies the full
OutRo method through `OutroController` in `outro.py`.

Both components from the paper are enabled together:

- **ReLU–tanh gated head output rotation** aligns non-sink head outputs with the sink value direction.
- **Sink information enhancement via mask relaxation** lets sink positions aggregate the full prompt.

The attention integration precedes `o_proj` and uses FlashAttention 2.
This copy depends on Transformers 4.52.3 internals.
