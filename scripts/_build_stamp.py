import hashlib
import io
import time

adapter = r"kaggle_datasets/vlanav_src/src/vlanav/model/omnivla_adapter.py"
env = r"kaggle_datasets/vlanav_src/src/vlanav/environment/ai2thor_env.py"

h = hashlib.sha256(open(adapter, "rb").read()).hexdigest()[:16]
a = io.open(adapter, encoding="utf-8").read()
e = io.open(env, encoding="utf-8").read()
lines = [
    "built_utc=" + time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "adapter_sha256_16=" + h,
    "contains_pad_sequence=" + str("pad_sequence" in a),
    "contains_detach=" + str("detach()" in a),
    "contains_RotateLeft=" + str("RotateLeft" in e),
    "contains_D_recovery_yaw=" + str("D_recovery" in e),
]
io.open(r"kaggle_datasets/vlanav_src/BUILD.txt", "w", encoding="utf-8").write(
    "\n".join(lines) + "\n")
print("\n".join(lines))
