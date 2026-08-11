# Generated on the host. No parsing, no arrays, no mapfile — three
# backfill runs died in shell list-building while the data file was
# provably correct, so the list IS the code now.

run_model "m06" "Qwen/Qwen3-1.7B-Base" "p1:world_v0:7301 p1:world_v0:9412 p1:world_v2:5150 p2:world_v2:5150 p2:world_v2:9412 p3:world_v0:7301 p3:world_v0:9412 p4:world_v0:9412 p4:world_v2:5150 p5:world_v0:5150 p5:world_v0:9412 p5:world_v2:7301 p5:world_v2:9412"
run_model "m08" "Qwen/Qwen3-4B-Base" "p3:world_v0:5150 p3:world_v0:9412 p4:world_v0:5150"
run_model "m10" "Qwen/Qwen3-8B-Base" "p1:world_v0:9412 p2:world_v2:7301 p3:world_v2:9412"
run_model "m15" "mistralai/Mistral-7B-Instruct-v0.3" "p1:world_v2:9412 p2:world_v0:5150 p2:world_v0:7301 p2:world_v0:9412 p2:world_v2:5150 p2:world_v2:7301 p2:world_v2:9412 p3:world_v0:5150 p3:world_v0:7301 p3:world_v0:9412 p3:world_v2:5150 p3:world_v2:7301 p3:world_v2:9412 p4:world_v0:5150 p4:world_v0:7301 p4:world_v0:9412 p4:world_v2:5150 p4:world_v2:7301 p4:world_v2:9412 p5:world_v0:5150"
