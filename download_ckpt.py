from huggingface_hub import snapshot_download

dataset_name = "LucasFang/PUMA"

local_dir = "./ckpts"

token = "<token>"  

snapshot_download(
    repo_id=dataset_name,
    repo_type="model",
    local_dir=local_dir,
    token=token,
    resume_download=True,
    local_dir_use_symlinks=False,
)
