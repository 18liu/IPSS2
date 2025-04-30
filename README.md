# Towards Scalable and Efficient Full-Reference Omnidirectional Image Quality Assessment
[Jiebin Yan], [Zhiyong Liu], [Zhihua Wang], [Yuming Fang], [Hantao Liu]

## Database:JUFE-10K

## :hammer_and_wrench: Usage

### ERP Patch Extraction
If you want to retrain the IPSS^2 model, using JUFE-10K database or another database, you first need to prepare ERP patch.
```
run sampling_alter.py
```
### Training IPSS^2
Modify the configuration in config.py
- Modify training and test dataset path
