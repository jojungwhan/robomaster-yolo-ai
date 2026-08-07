# Third-party model notice

## vsmidhun21/Lego-Identification

- Source: <https://github.com/vsmidhun21/Lego-Identification>
- Pinned commit: `ddd54ae077a8fed243065a1104ee14eb4aa5f5e2`
- Selected file: `FinalCoShSi.pt`
- SHA256: `87591257D011CC7409CFF14BABF28A1D15402AB521E75F3D10BF5F7A1E013CF6`
- Local destination: `models/lego-identification/FinalCoShSi.pt`

No license file was present in the repository at the pinned revision. The setup script
downloads the checkpoint directly for local evaluation; the checkpoint is ignored by
Git. Obtain permission from the repository author before redistributing it or building
on it for a public or commercial deployment.

The checkpoint was scanned with
`torch.serialization.get_unsafe_globals_in_checkpoint`. The reported globals were the
expected Ultralytics YOLOv8 detection modules and standard PyTorch layers. This is not a
proof that a pickle-based checkpoint is safe, so the application also verifies the
pinned SHA256 before loading the known filename.
