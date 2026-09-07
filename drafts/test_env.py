from dspy_factorio.env import get_environment_info, describe_env

print(describe_env("iron_ore_throughput"))
info = get_environment_info("iron_ore_throughput")
print(info)