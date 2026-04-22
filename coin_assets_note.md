# COIN Assets Note

## Current location

All transferred assets have been extracted under:

`/data0/sqx/coin_assets/`

The original archive files are still kept in the same directory next to the extracted folders.

## Extracted entries

- `Vicuna/`
- `clip-vit-large-patch14-336/`
- `llava_projectors/`
- `COCO2014/`
- `GQA/`
- `ImageNet_withlabel/`
- `OCR-VQA/`
- `RefCOCO/`
- `ScienceQA/`
- `TextVQA/`
- `VizWiz/`
- `playground/`

## Quick checks

Use these commands when you need to verify the layout or estimate space usage:

```bash
ls -lh /data0/sqx/coin_assets
du -sh /data0/sqx/coin_assets/*
```

## Notes for later AI operations

- Treat `/data0/sqx/coin_assets/` as the shared staging folder for these assets.
- Do not move or rename the extracted folders unless the next step explicitly requires it.
- The archive files can be reused if a clean re-extraction is needed.