"""Model locations on the Hugging Face Hub. Kept free of heavy imports so the CLI can use them."""
from __future__ import annotations

PERSON_BALL_REPO = "grdesignbuild/volleyball-person-ball-detector"
JERSEY_REPO = "grdesignbuild/volleyball-jersey-number-reader"
REF_BALL_WEIGHTS = "hf://Davidsv/volley-ref-ai/yolo_volleyball_ball.pt"
COURT_WEIGHTS = "hf://Davidsv/volley-ref-ai/yolo_court_keypoints.pt"


def resolve_weights(spec: str) -> str:
    """`hf://owner/repo/file.pt` -> local cached path; anything else is returned unchanged."""
    if not spec.startswith("hf://"):
        return spec
    from huggingface_hub import hf_hub_download
    owner, repo, filename = spec[len("hf://"):].split("/", 2)
    return hf_hub_download(f"{owner}/{repo}", filename)
