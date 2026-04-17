# Examples

Copy-paste recipes that exercise BODHI's core capabilities.

| File | What |
|---|---|
| `01_chat_basics.py` | Programmatic chat turn with brain-state readback |
| `02_teach_from_image.py` | Runtime concept acquisition — compute a fingerprint from your own image |
| `03_nightly_train.sh` | LoRA fine-tune on your accumulated conversations |
| `04_inspect_brain_state.py` | Dump the raw `brain_result` dict: bands, worm, regions, groups |
| `05_sleep_and_dreams.py` | Watch Hebbian growth + triangle inference across 5 sleep cycles |

Run each from the project root, e.g. `python examples/01_chat_basics.py`.

For the interactive browser UI: `python chat_ui.py` (needs `pip install flask`).
