# Generated on the host. The list IS the code -- see gpu_job31's header.
#
# PROBE SCOPE: m08 and m10 only. m15 was already recovered by job31 (its
# cells were push-lost, and the batching fix worked). m06 is deliberately
# EXCLUDED: 13 cells on the worst offender (46.9 chars/cmd), and even a
# full recovery leaves it at n~65, under the n>=100 floor. This probe
# tests whether the raised timeout fixes the eval failure at all.

run_model "m08" "Qwen/Qwen3-4B-Base" "p3:world_v0:5150 p3:world_v0:9412 p4:world_v0:5150"
run_model "m10" "Qwen/Qwen3-8B-Base" "p1:world_v0:9412 p2:world_v2:7301 p3:world_v2:9412"
