"""
Compare original vs tiny model loading
"""
from transformers import AutoConfig, AutoModelForCausalLM

print("="*70)
print("Testing Original Model (160MB)")
print("="*70)
try:
    config_orig = AutoConfig.from_pretrained(
        "optimum-intel-internal-testing/tiny-random-MiniCPM-o-2_6",
        trust_remote_code=True
    )
    print(f"✅ Original config loaded: {config_orig.model_type}")
    
    model_orig = AutoModelForCausalLM.from_pretrained(
        "optimum-intel-internal-testing/tiny-random-MiniCPM-o-2_6",
        trust_remote_code=True
    )
    print(f"✅ Original model loaded: {sum(p.numel() for p in model_orig.parameters()):,} params")
except Exception as e:
    print(f"❌ Original model failed: {e}")

print("\n" + "="*70)
print("Testing Tiny Model (0.4MB)")
print("="*70)
try:
    config_tiny = AutoConfig.from_pretrained(
        "M-Ziyo/tiny-random-MiniCPM-o-2_6",
        trust_remote_code=True
    )
    print(f"✅ Tiny config loaded: {config_tiny.model_type}")
    
    model_tiny = AutoModelForCausalLM.from_pretrained(
        "M-Ziyo/tiny-random-MiniCPM-o-2_6",
        trust_remote_code=True
    )
    print(f"✅ Tiny model loaded: {sum(p.numel() for p in model_tiny.parameters()):,} params")
except Exception as e:
    print(f"❌ Tiny model failed: {e}")

print("\n" + "="*70)
print("Comparison Complete")
print("="*70)

