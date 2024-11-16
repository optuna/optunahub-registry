import optunahub


module = optunahub.load_module(
    package="optuna_enhanced_visualization",
    repo_owner="nourbc211",
    ref="main",
)

print("Module loaded successfully:", module)
