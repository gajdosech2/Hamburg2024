- export CUDA_VISIBLE_DEVICES="0"
- ssh -L 17007:127.0.0.1:7007 user@host
- localhost:17007/
- du -a -h --max-depth=1 | sort -hr
  
label-studio-converter import coco\
  -i coco_annotations.json \
  -o label_studio_annotations.json

- export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
- export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/home/g/gajdosech2/Hamburg2024
- ssh -L 18080:127.0.0.1:8080 uran
