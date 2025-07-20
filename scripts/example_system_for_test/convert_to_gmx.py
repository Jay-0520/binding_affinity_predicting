from pathlib import Path

import parmed as pmd

input_dir = Path(
    "/Users/jingjinghuang/Documents/fep_workflow/binding_affinity_predicting/scripts/some_utils_code"  # noqa: E501
)
prm7_files = list(input_dir.glob("*.prm7"))

for prm7 in prm7_files:
    rst7 = prm7.with_suffix(".rst7")
    if not rst7.exists():
        print(f"❌ Skipping {prm7.name}: matching .rst7 not found")
        continue

    print(f"🔄 Converting {prm7.name} + {rst7.name}")
    structure = pmd.load_file(str(prm7), str(rst7))

    out_prefix = prm7.stem
    # structure.save(str(input_dir / f"{out_prefix}.top"), format="gromacs")
    structure.save(str(input_dir / f"{out_prefix}.gro"), format="gro")

    print(f"✅ Saved: {out_prefix}.gro")
