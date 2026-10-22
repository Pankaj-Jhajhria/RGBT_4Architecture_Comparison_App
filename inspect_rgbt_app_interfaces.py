from pathlib import Path
import ast
import torch

ROOT = Path(r"C:\Users\panka\Downloads\AI_Engineer\Projects\College\MultiModel_Segmentation\RGBT_4Architecture_Comparison_App")
TARGETS = [
    ("MFNet", "architectures/mfnet/model.py", "MFNet", ["__init__", "forward"], ["build_model"]),
    ("CSFNet", "architectures/csfnet/models/csfnet.py", "CSFNet", ["__init__", "forward"], ["build_csfnet"]),
    ("CMAF", "architectures/cmaf/my_model.py", "CustomDualStreamUNet", ["__init__", "forward"], ["build_model"]),
    ("RTFNet18V2", "architectures/rtfnet18v2/model19_v2.py", "RTFNet18V2", ["__init__", "forward"], ["build_model"]),
]

def print_node(path, text, node, max_lines=42):
    lines = text.splitlines()
    start = node.lineno
    end = getattr(node, "end_lineno", start)
    print(f"  {node.name} signature: {node.name}({ast.unparse(node.args)})  [lines {start}-{end}]")
    shown_end = min(end, start + max_lines - 1)
    for i in range(start, shown_end + 1):
        print(f"    {i:4}: {lines[i-1]}")
    if shown_end < end:
        print(f"    ... ({end-shown_end} lines omitted)")

print("="*78)
print("MODEL INTERFACES")
print("="*78)
for label, rel, classname, methods, funcs in TARGETS:
    path = ROOT / rel
    print(f"\n### {label}: {rel}")
    if not path.is_file():
        print("MISSING SOURCE", path)
        continue
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        print("PARSE ERROR", exc)
        continue
    for n in tree.body:
        if isinstance(n, ast.ClassDef) and n.name == classname:
            for child in n.body:
                if isinstance(child, ast.FunctionDef) and child.name in methods:
                    print_node(path, text, child, 38 if child.name == "__init__" else 48)
        elif isinstance(n, ast.FunctionDef) and n.name in funcs:
            print_node(path, text, n, 24)

print("\n" + "="*78)
print("CHECKPOINT CONFIGURATION / SAVED METRICS")
print("="*78)
checkpoints = [
    ("MFNet", "weights/mfnet_best_checkpoint.pt", "model_state_dict"),
    ("CSFNet-2", "weights/csfnet2_best_model.pth", "model_state_dict"),
    ("CMAF", "weights/cmaf_best_checkpoint.pt", "model_state_dict"),
    ("RTFNet18V2", "weights/rtfnet18v2_best_fp16.pt", "model"),
]
for label, rel, statekey in checkpoints:
    path = ROOT / rel
    print(f"\n### {label}: {rel}")
    if not path.is_file():
        print("MISSING CHECKPOINT", path)
        continue
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        print("LOAD ERROR", type(exc).__name__, exc)
        continue
    print("top-level keys:", list(ck.keys()) if isinstance(ck, dict) else type(ck).__name__)
    print("state key exists:", isinstance(ck, dict) and statekey in ck)
    for k in ("epoch", "best_epoch", "best_metric_val", "best_score", "select_metric"):
        if isinstance(ck, dict) and k in ck:
            print(f"{k}: {str(ck[k])[:300]}")
    for k in ("config_snapshot", "metrics", "args", "model_info", "val_metrics"):
        value = ck.get(k) if isinstance(ck, dict) else None
        if isinstance(value, dict):
            print(f"{k} keys:", list(value.keys())[:80])
            if k == "config_snapshot":
                for kk, vv in value.items():
                    if any(word in kk.lower() for word in ("input", "image", "thermal", "rgb", "mean", "std", "size", "class", "backbone", "pretrain", "norm", "width", "height", "fusion", "model", "data")):
                        print(f"  {kk}: {str(vv)[:180]}")
            elif k == "args":
                for kk in ("img_size", "head", "thermal_norm", "rgb_norm", "class0_loss", "select_metric", "data_root", "pretrained", "thermal_init", "bn_mode"):
                    if kk in value:
                        print(f"  {kk}: {str(value[kk])[:180]}")
            elif k == "model_info":
                for kk, vv in value.items():
                    print(f"  {kk}: {str(vv)[:220]}")
            elif k == "metrics":
                for kk, vv in value.items():
                    print(f"  {kk}: {str(vv)[:180]}")
            elif k == "val_metrics":
                for kk in ("miou_all9", "miou_fg8", "pixel_acc", "mean_dice_all9", "mean_dice_fg8"):
                    if kk in value:
                        print(f"  {kk}: {value[kk]}")
        elif value is not None:
            print(f"{k}: {str(value)[:300]}")
print("\nInspection complete.")
